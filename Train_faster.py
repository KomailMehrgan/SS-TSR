import argparse
import numpy as np
import pandas as pd
import string
import os
from datetime import datetime
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, random_split, Subset, Dataset, ConcatDataset
from torch.amp import autocast, GradScaler
from collections import OrderedDict
from torchvision import transforms
import torchvision.transforms.functional as F

# --- Model Imports ---
from model.tsrn import TSRN
from Network.srresnet import _NetG as SRResNet
from model.rdn import RDN
from model.srcnn import SRCNN

# --- Helper Imports ---
from OCR.model import ModelOCR
from OCR.utils import AttnLabelConverter
from OCR.ocr_loss import OCRProcessor
from datasets.data_set_from_pt import DatasetFromPT


# --- Data Augmentation Classes ---
class JointRandomAugment:
    """
    Applies the same random augmentation to a pair of low-res and high-res images.
    Safe for OCR-based training: does not corrupt text.
    """

    def __init__(self, max_rotation=5, max_translation=2, scale_range=(0.95, 1.05),
                 brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.9, 1.1), gamma=(0.9, 1.1),
                 swap_prob=0.5):
        self.max_rotation = max_rotation
        self.max_translation = max_translation
        self.scale_range = scale_range
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.gamma = gamma
        self.swap_prob = swap_prob

    def __call__(self, lr_img, hr_img):
        # ---- Geometric transforms ----
        angle = random.uniform(-self.max_rotation, self.max_rotation)
        translate_x = random.uniform(-self.max_translation, self.max_translation)
        translate_y = random.uniform(-self.max_translation, self.max_translation)
        scale = random.uniform(*self.scale_range)

        lr_aug = F.affine(lr_img, angle=angle, translate=(translate_x, translate_y),
                          scale=scale, shear=0, interpolation=F.InterpolationMode.BILINEAR)
        hr_aug = F.affine(hr_img, angle=angle, translate=(translate_x, translate_y),
                          scale=scale, shear=0, interpolation=F.InterpolationMode.BILINEAR)

        # ---- Color / photometric transforms ----
        b = random.uniform(*self.brightness)
        c = random.uniform(*self.contrast)
        s = random.uniform(*self.saturation)
        g = random.uniform(*self.gamma)

        lr_rgb = lr_aug[:3, :, :]
        hr_rgb = hr_aug

        lr_rgb = F.adjust_brightness(lr_rgb, b)
        hr_rgb = F.adjust_brightness(hr_rgb, b)

        lr_rgb = F.adjust_contrast(lr_rgb, c)
        hr_rgb = F.adjust_contrast(hr_rgb, c)

        lr_rgb = F.adjust_saturation(lr_rgb, s)
        hr_rgb = F.adjust_saturation(hr_rgb, s)

        lr_rgb = F.adjust_gamma(lr_rgb, g)
        hr_rgb = F.adjust_gamma(hr_rgb, g)

        # ---- Optional RGB swap ----
        if random.random() < self.swap_prob:
            lr_rgb = lr_rgb[[2, 1, 0], :, :]
            hr_rgb = hr_rgb[[2, 1, 0], :, :]

        lr_aug[:3, :, :] = lr_rgb
        return lr_aug, hr_rgb

class AugmentedDataset(Dataset):
    """
    A wrapper dataset that applies joint random augmentations on the fly.
    """

    def __init__(self, original_dataset):
        self.original_dataset = original_dataset

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, index):
        input_img, real_img, ocr_label = self.original_dataset[index]
        augmenter = JointRandomAugment()
        aug_input, aug_real = augmenter(input_img, real_img)
        return aug_input, aug_real, ocr_label


def get_model(arch_name, device):
    """Initializes and returns the model and its creation parameters."""
    arch_name = arch_name.lower()
    print(f"===> Building model: {arch_name.upper()}")

    params = {}

    if arch_name == 'tsrn':
        params = {
            'scale_factor': 2, 'width': 128, 'height': 32,
            'STN': True, 'srb_nums': 24, 'mask': True, 'hidden_units': 96
        }
        model = TSRN(**params)
    elif arch_name == 'srresnet':
        params = {'num_channels': 64}
        model = SRResNet(**params)
    elif arch_name == 'rdn':
        params = {'scale_factor': 2}
        model = RDN(**params)
    elif arch_name == 'srcnn':
        params = {'scale_factor': 2}
        model = SRCNN(**params)
    else:
        raise ValueError(f"Unknown architecture: {arch_name}")

    return model.to(device), params


def save_run_details(run_folder, opt, model, model_params, full_dataset, train_dataset, val_dataset,
                     current_ocr_weight):
    """Saves a comprehensive summary of the training run to a text file."""
    summary_path = os.path.join(run_folder, 'run_summary.txt')

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    with open(summary_path, 'w') as f:
        f.write("================ Training Run Summary ================\n\n")

        f.write("--- Command-Line Arguments ---\n")
        for key, value in sorted(vars(opt).items()):
            f.write(f"{key:<22}: {value}\n")
        f.write("\n")

        f.write("--- Run-Specific Parameters ---\n")
        f.write(f"{'current_ocr_weight':<22}: {current_ocr_weight}\n")
        f.write("\n")

        f.write("--- Dataset Information ---\n")
        f.write(f"{'Full dataset size':<22}: {len(full_dataset)} samples\n")
        f.write(f"{'Training samples':<22}: {len(train_dataset)} (after augmentation)\n")
        f.write(f"{'Validation samples':<22}: {len(val_dataset)}\n")
        f.write("\n")

        f.write("--- Model Details ---\n")
        f.write(f"{'Architecture':<22}: {opt.arch.upper()}\n")
        f.write(f"{'Total parameters':<22}: {total_params:,}\n")
        f.write(f"{'Trainable params':<22}: {trainable_params:,}\n")
        f.write("\n")

        f.write("--- Model Hyperparameters ---\n")
        if model_params:
            for key, value in model_params.items():
                f.write(f"{key:<22}: {value}\n")
        else:
            f.write("No specific hyperparameters recorded.\n")
        f.write("\n")

        f.write("--- Optimizer & Scheduler ---\n")
        f.write(f"{'Optimizer':<22}: AdamW (fused)\n")
        f.write(f"{'Learning Rate (lr)':<22}: {opt.lr}\n")
        f.write(f"{'Weight Decay':<22}: 1e-5 (hardcoded)\n")
        f.write(f"{'LR Scheduler':<22}: StepLR\n")
        f.write(f"{'LR Step Size':<22}: {opt.step}\n")
        f.write(f"{'LR Decay Gamma':<22}: 0.1 (hardcoded)\n")
        f.write("\n")

        f.write("--- System Information ---\n")
        f.write(f"{'PyTorch Version':<22}: {torch.__version__}\n")
        if torch.cuda.is_available():
            f.write(f"{'CUDA Version':<22}: {torch.version.cuda}\n")
            f.write(f"{'GPU':<22}: {torch.cuda.get_device_name(0)}\n")
        else:
            f.write(f"{'Device':<22}: CPU\n")

    print(f"Comprehensive run summary saved to {summary_path}")


def main():
    """Main function to parse arguments and orchestrate the training process."""
    parser = argparse.ArgumentParser(
        description="Optimized SR Training Script with OCR supervision and Gradient Accumulation")
    # Arguments...
    parser.add_argument('--arch', type=str, default="tsrn", choices=['tsrn', 'srresnet', 'rdn', 'srcnn'])
    parser.add_argument("--dataset", type=str, default="datasets/dataset.pt")
    parser.add_argument('--ocr_weight', type=float, default=0.01)
    parser.add_argument('--ablation_weights', default=[0.005,0.0001,0.001,0.0005], type=float, nargs='+')
    parser.add_argument("--scale", type=float, default=0.02)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--batchSize", type=int, default=32)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--aug", type=int, default=2)
    parser.add_argument("--nEpochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--step", type=int, default=50)
    parser.add_argument("--cuda", action="store_false")
    parser.add_argument("--gpus", default="0", type=str)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--start-epoch", default=1, type=int)
    opt = parser.parse_args()
    print(opt)

    # --- Device setup ---
    device = torch.device("cuda" if opt.cuda and torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        print(f"Using GPU(s): {opt.gpus}")
    else:
        print("Using CPU.")
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed(42)
    cudnn.benchmark = True

    # --- Dataset ---
    print("===> Loading dataset...")
    full_dataset = DatasetFromPT(opt.dataset)
    total_size = len(full_dataset)
    subset_size = int(total_size * opt.scale)
    indices = list(range(total_size))[:subset_size]
    study_subset = Subset(full_dataset, indices)
    val_size = int(opt.val_split * len(study_subset))
    train_size = len(study_subset) - val_size
    train_dataset = Subset(study_subset, range(train_size))
    val_dataset = Subset(study_subset, range(train_size, len(study_subset)))

    # --- Augmentation Handling ---
    # --- Augmentation Handling ---
    if opt.aug > 0:
        print("\n===> Using online augmentation (no dataset size increase).")
        train_dataset = AugmentedDataset(train_dataset)
    else:
        print("\n===> Augmentation disabled.")
        train_dataset = train_dataset

    train_loader = DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.threads,
                              pin_memory=True, drop_last=True, persistent_workers=(opt.threads > 0))
    val_loader = DataLoader(val_dataset, batch_size=opt.batchSize, shuffle=False, num_workers=opt.threads,
                            pin_memory=True, persistent_workers=(opt.threads > 0))

    # --- OCR Model Setup ---
    print("===> Setting up OCR model...")
    from argparse import Namespace
    ocr_config = Namespace(Transformation='TPS', FeatureExtraction='ResNet', SequenceModeling='BiLSTM',
                           Prediction='Attn', num_fiducial=20, input_channel=1, output_channel=512,
                           hidden_size=256, num_class=96, imgH=32, imgW=100, batch_max_length=35, device=device)
    netOCR = ModelOCR(ocr_config).to(device)
    state_dict = torch.load("back_up_models/OCR/TPS-ResNet-BiLSTM-Attn-case-sensitive.pth", map_location=device,
                            weights_only=True)
    new_state_dict = OrderedDict((k[7:] if k.startswith('module.') else k, v) for k, v in state_dict.items())
    netOCR.load_state_dict(new_state_dict)
    for p in netOCR.parameters():
        p.requires_grad = False
    netOCR.train()
    for module in netOCR.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()
    character = string.printable[:-6]
    converter = AttnLabelConverter(character)
    ocr_processor = OCRProcessor(netOCR, converter, device)
    mse_criterion = nn.MSELoss(reduction='mean').to(device)

    # --- Training Loop ---
    weights_to_run = opt.ablation_weights if opt.ablation_weights else [opt.ocr_weight]
    start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    parent_folder = os.path.join("experiments", f"{opt.arch}_{start_time}")
    os.makedirs(parent_folder, exist_ok=True)
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    for ocr_weight in weights_to_run:
        run_id = f"weight_{ocr_weight}" if opt.ablation_weights else "run"
        run_folder = os.path.join(parent_folder, run_id)
        os.makedirs(run_folder, exist_ok=True)

        torch.manual_seed(42)
        netSR, model_params = get_model(opt.arch, device)

        # Save comprehensive run details after model is created
        save_run_details(run_folder, opt, netSR, model_params, full_dataset, train_dataset, val_dataset, ocr_weight)

        if int(torch.__version__.split('.')[0]) >= 2:
            netSR = torch.compile(netSR)

        optimizer = torch.optim.AdamW(netSR.parameters(), lr=opt.lr, weight_decay=1e-5, fused=(device.type == 'cuda'))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.step, gamma=0.1)
        metrics = {'epoch': [], 'train_img_loss': [], 'train_ocr_loss': [], 'val_img_loss': [], 'val_ocr_loss': []}

        for epoch in range(opt.start_epoch, opt.nEpochs + 1):
            train_losses = train_one_epoch(opt, train_loader, netSR, ocr_processor, mse_criterion,
                                           optimizer, epoch, device, ocr_weight, scaler)
            val_losses = validate_one_epoch(opt, val_loader, netSR, ocr_processor, mse_criterion, epoch, device)
            scheduler.step()
            metrics['epoch'].append(epoch)
            metrics['train_img_loss'].append(train_losses['img'])
            metrics['train_ocr_loss'].append(train_losses['ocr'])
            metrics['val_img_loss'].append(val_losses['img'])
            metrics['val_ocr_loss'].append(val_losses['ocr'])
            save_and_plot_metrics(metrics, run_folder)
            if epoch % 50 == 0:
                save_checkpoint(netSR, epoch, run_folder)


def train_one_epoch(opt, data_loader, netSR, ocr_processor, mse_criterion, optimizer, epoch, device, ocr_weight,
                    scaler):
    """Trains the model for one epoch with gradient accumulation."""
    netSR.train()
    img_losses, ocr_losses = [], []
    optimizer.zero_grad()
    for iter_idx, batch in enumerate(data_loader, 1):
        input_img, real_img, ocr_label = batch
        input_img, real_img = input_img.to(device), real_img.to(device)
        with autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            model_input = input_img if opt.arch == 'tsrn' else input_img[:, :3, :, :]
            sr_image = netSR(model_input)
            sr_image_rgb = sr_image[:, :3, :, :] if sr_image.shape[1] >= 3 else sr_image
            img_loss = mse_criterion(real_img, sr_image_rgb)
            sr_gray = sr_image_rgb.mean(dim=1, keepdim=True)
            ocr_loss = ocr_processor.process(sr_gray, ocr_label)
            total_loss = img_loss + ocr_weight * ocr_loss
            if opt.accumulation_steps > 1:
                total_loss = total_loss / opt.accumulation_steps
        scaler.scale(total_loss).backward()
        if (iter_idx % opt.accumulation_steps == 0) or (iter_idx == len(data_loader)):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(netSR.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        img_losses.append(img_loss.item())
        ocr_losses.append(ocr_loss.item())
    print(
        f"--- Epoch {epoch} Training Summary --- Img Loss: {np.mean(img_losses):.4f}, OCR Loss: {np.mean(ocr_losses):.4f}")
    return {'img': np.mean(img_losses), 'ocr': np.mean(ocr_losses)}


def validate_one_epoch(opt, data_loader, netSR, ocr_processor, mse_criterion, epoch, device):
    """Validates the model for one epoch."""
    netSR.eval()
    img_losses, ocr_losses = [], []
    with torch.no_grad():
        for batch in data_loader:
            input_img, real_img, ocr_label = batch
            input_img, real_img = input_img.to(device), real_img.to(device)
            with autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                model_input = input_img if opt.arch == 'tsrn' else input_img[:, :3, :, :]
                sr_image = netSR(model_input)
                sr_image_rgb = sr_image[:, :3, :, :] if sr_image.shape[1] >= 3 else sr_image
                img_loss = mse_criterion(real_img, sr_image_rgb)
                sr_gray = sr_image_rgb.mean(dim=1, keepdim=True)
                ocr_loss = ocr_processor.process(sr_gray, ocr_label)
            img_losses.append(img_loss.item())
            ocr_losses.append(ocr_loss.item())
    print(
        f"--- Epoch {epoch} Validation Summary --- Img Loss: {np.mean(img_losses):.4f}, OCR Loss: {np.mean(ocr_losses):.4f}")
    return {'img': np.mean(img_losses), 'ocr': np.mean(ocr_losses)}


def save_checkpoint(model, epoch, save_folder):
    """Saves a model checkpoint, handling torch.compile wrapper."""
    os.makedirs(os.path.join(save_folder, "checkpoints"), exist_ok=True)
    model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
    path = os.path.join(save_folder, "checkpoints", f"model_epoch_{epoch}.pth")
    torch.save({"epoch": epoch, "model_state_dict": model_to_save.state_dict()}, path)
    print(f"Checkpoint saved: {path}")


def save_and_plot_metrics(metrics, save_folder):
    """Saves metrics to a CSV and plots the training/validation loss curves."""
    df = pd.DataFrame(metrics)
    df.to_csv(os.path.join(save_folder, "metrics.csv"), index=False)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    axs[0].plot(df['epoch'], df['train_img_loss'], 'o-', label='Train Image Loss')
    axs[0].plot(df['epoch'], df['val_img_loss'], 'o-', label='Val Image Loss')
    axs[0].set_title('MSE Image Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[1].plot(df['epoch'], df['train_ocr_loss'], 'o-', label='Train OCR Loss')
    axs[1].plot(df['epoch'], df['val_ocr_loss'], 'o-', label='Val OCR Loss')
    axs[1].set_title('OCR Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].legend()
    fig.suptitle(f"Metrics for {os.path.basename(save_folder)}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(save_folder, "loss_plot.png"))
    plt.close()


if __name__ == "__main__":
    main()