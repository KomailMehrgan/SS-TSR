import argparse
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset, Dataset, ConcatDataset
from collections import OrderedDict
import os
import torch
import pandas as pd
import string
from datetime import datetime
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import random
from torchvision import transforms  # Added for easier access to InterpolationMode

# --- Model Imports ---
from model.tsrn import TSRN
from Network.srresnet import _NetG as SRResNet
from model.rdn import RDN
from model.srcnn import SRCNN

# --- Helper Imports ---
from OCR.model import ModelOCR
from OCR.utils import AttnLabelConverter
from OCR.ocr_loss import OCRProcessor
from datasets.data_set_from_pt import DatasetFromPT  # Using the efficient .pt loader


# --- Data Augmentation Classes ---
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
    print(f"===> Building model: {arch_name.upper()}")
    arch_name = arch_name.lower()

    params = {}

    if arch_name == 'tsrn':
        params = {
            'scale_factor': 2, 'width': 128, 'height': 32,
            'STN': True, 'srb_nums': 12, 'mask': True, 'hidden_units': 64
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
        raise ValueError(f"Unknown architecture specified: {arch_name}")

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
        f.write(f"{'Optimizer':<22}: AdamW\n")
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
    parser = argparse.ArgumentParser(description="Comprehensive Training and Ablation Study Script for SR Models")
    # Arguments...
    parser.add_argument('--arch', type=str, default="tsrn", choices=['tsrn', 'srresnet', 'rdn', 'srcnn'])
    parser.add_argument("--dataset", type=str, default="datasets/dataset.pt")
    parser.add_argument('--ocr_weight', type=float, default=0.01)
    parser.add_argument('--ablation_weights', default=[0.001], type=float, nargs='+')
    parser.add_argument("--scale", type=float, default=0.03)
    parser.add_argument("--val_split", type=float, default=0)
    parser.add_argument("--batchSize", type=int, default=4)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--aug", type=int, default=0)
    parser.add_argument("--nEpochs", type=int, default=350)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--step", type=int, default=50)
    parser.add_argument("--cuda", action="store_false")
    parser.add_argument("--gpus", default="0", type=str)
    parser.add_argument("--resume", default="back_up_models/SR/ss-tsrn/ss-tsrn_300.pth", type=str)
    parser.add_argument("--start-epoch", default=301, type=int)
    opt = parser.parse_args()

    print("--- Unified Training Script ---")
    print(opt)

    # --- Setup Device and Reproducibility ---
    device = torch.device("cuda" if opt.cuda and torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using GPU(s): {opt.gpus}")
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    else:
        print("Using CPU.")
    reproducibility_seed = 42
    torch.manual_seed(reproducibility_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(reproducibility_seed)
    cudnn.benchmark = True

    # --- Dataset Loading and Splitting ---
    print("\n===> Loading preprocessed dataset...")
    full_dataset = DatasetFromPT(opt.dataset)
    total_size = len(full_dataset)
    subset_size = int(total_size * opt.scale)
    indices = list(range(total_size))[:subset_size]
    study_subset = Subset(full_dataset, indices)
    print(f"Using a subset of {len(study_subset)} images ({opt.scale * 100:.1f}% of total).")
    val_size = int(opt.val_split * len(study_subset))
    train_size = len(study_subset) - val_size
    train_dataset = Subset(study_subset, range(train_size))
    val_dataset = Subset(study_subset, range(train_size, len(study_subset)))
    print(f"Split into {len(train_dataset)} training and {len(val_dataset)} validation images.")

    # --- Augmentation Handling ---
    if opt.aug > 0:
        print("\n===> Using online augmentation (no dataset size increase).")
        train_dataset = AugmentedDataset(train_dataset)
    else:
        print("\n===> Augmentation disabled.")
        train_dataset = train_dataset

    train_loader = DataLoader(dataset=train_dataset, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True,
                              drop_last=True, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=False,
                            pin_memory=True)

    # --- OCR Model Setup ---
    print("\n===> Building and loading OCR model...")
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
    # character='0123456789abcdefghijklmnopqrstuvwxyz'
    character = string.printable[:-6]
    converter = AttnLabelConverter(character)
    ocr_processor = OCRProcessor(netOCR, converter, device)
    # mse_criterion = nn.MSELoss(reduction='mean').to(device)
    mse_criterion = nn.L1Loss(reduction='mean').to(device)

    # --- Determine Training Mode ---
    weights_to_run = opt.ablation_weights if opt.ablation_weights is not None else [opt.ocr_weight]
    print(f"\n--- Starting Ablation Study for weights: {weights_to_run} ---")

    # --- Main Training Loop ---
    start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    parent_folder = os.path.join("experiments", f"{opt.arch}_{start_time}")
    os.makedirs(parent_folder, exist_ok=True)
    print(f"Results will be saved in: {parent_folder}")

    for ocr_weight in weights_to_run:
        run_identifier = f"weight_{ocr_weight}" if opt.ablation_weights is not None else "run"
        print(f"\n{'=' * 25} RUNNING: {run_identifier} {'=' * 25}")
        run_folder = os.path.join(parent_folder, run_identifier)
        os.makedirs(run_folder, exist_ok=True)

        torch.manual_seed(reproducibility_seed)
        netSR, model_params = get_model(opt.arch, device)

        # =========================================================================
        # 1. NEW: RESUME LOGIC ADDED HERE
        # =========================================================================
        current_start_epoch = opt.start_epoch  # Default start

        if opt.resume and opt.resume.strip():
            if os.path.isfile(opt.resume):
                print(f"===> Resuming from checkpoint: {opt.resume}")
                checkpoint = torch.load(opt.resume, map_location='cpu', weights_only=False)
                netSR.load_state_dict(checkpoint['model'].state_dict())


        # Save comprehensive run details
        save_run_details(run_folder, opt, netSR, model_params, full_dataset, train_dataset, val_dataset, ocr_weight)

        optimizer = torch.optim.AdamW(netSR.parameters(), lr=opt.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.step, gamma=0.5)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=0)
        metrics = {'epoch': [], 'train_img_loss': [], 'train_ocr_loss': [], 'val_img_loss': [], 'val_ocr_loss': []}

        for epoch in range(opt.start_epoch, opt.nEpochs + 1):
            train_losses = train_one_epoch(opt, train_loader, netSR, ocr_processor, mse_criterion, optimizer, epoch,
                                           device, ocr_weight, opt.accumulation)
            val_losses = validate_one_epoch(opt, val_loader, netSR, ocr_processor, mse_criterion, epoch, device,
                                            ocr_weight)
            scheduler.step()
            metrics['epoch'].append(epoch)
            metrics['train_img_loss'].append(train_losses['img'])
            metrics['train_ocr_loss'].append(train_losses['ocr'])
            metrics['val_img_loss'].append(val_losses['img'])
            metrics['val_ocr_loss'].append(val_losses['ocr'])
            save_and_plot_metrics(metrics, run_folder)
            if epoch % 50 == 0:
                save_checkpoint(netSR, epoch, run_folder)

        print(f"{'=' * 25} FINISHED: {run_identifier} {'=' * 25}")


def train_one_epoch(opt, data_loader, netSR, ocr_processor, mse_criterion, optimizer, epoch, device, ocr_weight,
                    accumulation_steps=1):
    netSR.train()
    img_loss_list, ocr_loss_list = [], []
    optimizer.zero_grad()
    for iteration, batch in enumerate(data_loader, 1):
        input_img, real_img, ocr_label = batch
        input_img, real_img = input_img.to(device), real_img.to(device)
        model_input = input_img if opt.arch == 'tsrn' else input_img[:, :3, :, :]
        sr_image = netSR(model_input)
        sr_image_rgb = sr_image[:, :3, :, :] if sr_image.shape[1] == 4 else sr_image
        img_loss = mse_criterion(real_img, sr_image_rgb)
        sr_image_gray = sr_image_rgb.mean(dim=1, keepdim=True)
        ocr_loss = ocr_processor.process(sr_image_gray, ocr_label)
        total_loss = (img_loss * 1.0 + ocr_loss * ocr_weight) / accumulation_steps


        total_loss.backward()
        img_loss_list.append(img_loss.item())
        ocr_loss_list.append(ocr_loss.item())
        if iteration % accumulation_steps == 0 or iteration == len(data_loader):
            # torch.nn.utils.clip_grad_norm_(netSR.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
    avg_losses = {'img': np.mean(img_loss_list), 'ocr': np.mean(ocr_loss_list)}
    print(
        f"--- Epoch {epoch} Training Summary --- Img Loss: {avg_losses['img']:.4f}, OCR Loss: {avg_losses['ocr']:.4f}")
    return avg_losses


def validate_one_epoch(opt, data_loader, netSR, ocr_processor, mse_criterion, epoch, device, ocr_weight):
    netSR.eval()

    if len(data_loader) == 0:
        print(f"--- Epoch {epoch} Validation Summary --- Skipped (No validation data)")
        return {'img': 0.0, 'ocr': 0.0}
        # -------------------

    img_loss_list, ocr_loss_list = [], []
    with torch.no_grad():
        for batch in data_loader:
            input_img, real_img, ocr_label = batch
            input_img, real_img = input_img.to(device), real_img.to(device)
            model_input = input_img if opt.arch == 'tsrn' else input_img[:, :3, :, :]
            sr_image = netSR(model_input)
            sr_image_rgb = sr_image[:, :3, :, :] if sr_image.shape[1] == 4 else sr_image
            img_loss = mse_criterion(real_img, sr_image_rgb)
            sr_image_gray = sr_image_rgb.mean(dim=1, keepdim=True)
            ocr_loss = ocr_processor.process(sr_image_gray, ocr_label)
            img_loss_list.append(img_loss.item())
            ocr_loss_list.append(ocr_loss.item())
    avg_losses = {'img': np.mean(img_loss_list), 'ocr': np.mean(ocr_loss_list)}
    print(
        f"--- Epoch {epoch} Validation Summary --- Img Loss: {avg_losses['img']:.4f}, OCR Loss: {avg_losses['ocr']:.4f}")
    return avg_losses


def save_checkpoint(model, epoch, save_folder):
    checkpoint_dir = os.path.join(save_folder, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_out_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pth")
    state = {"epoch": epoch, "model_state_dict": model.state_dict()}
    torch.save(state, model_out_path)
    print(f"Checkpoint saved to {model_out_path}")


def save_and_plot_metrics(metrics, save_folder):
    csv_path = os.path.join(save_folder, 'metrics.csv')
    plot_path = os.path.join(save_folder, 'loss_plot.png')
    df = pd.DataFrame(metrics)
    df.to_csv(csv_path, index=False)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    axs[0].plot(df['epoch'], df['train_img_loss'], 'o-', label='Train Image Loss')
    axs[0].plot(df['epoch'], df['val_img_loss'], 'o-', label='Validation Image Loss')
    axs[0].set_title('Image Reconstruction Loss (MSE)')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[1].plot(df['epoch'], df['train_ocr_loss'], 'o-', label='Train OCR Loss')
    axs[1].plot(df['epoch'], df['val_ocr_loss'], 'o-', label='Validation OCR Loss')
    axs[1].set_title('OCR Attention Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].legend()
    fig.suptitle(f"Metrics for {os.path.basename(save_folder)}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(plot_path)
    plt.close()


if __name__ == "__main__":
    main()