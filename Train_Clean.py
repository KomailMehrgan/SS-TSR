import os
import random
import string
import argparse
from datetime import datetime
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader, Subset, Dataset

# --- Custom Model Imports ---
from model.tsrn import TSRN
from Network.srresnet import _NetG as SRResNet
from model.rdn import RDN
from model.srcnn import SRCNN

# --- OCR Imports ---
from OCR.model import ModelOCR
from OCR.utils import AttnLabelConverter
from OCR.ocr_loss import OCRProcessor
from datasets.data_set_from_pt import DatasetFromPT

# --- Constants ---
REPRODUCIBILITY_SEED = 42


# =========================================================================
# 1. CONFIGURATION & SETUP (GO FIRST)
# =========================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(description="SR Model Training with OCR Loss")
    parser.add_argument('--arch', type=str, default="tsrn", choices=['tsrn', 'srresnet', 'rdn', 'srcnn'])
    parser.add_argument("--dataset", type=str, default="datasets/dataset.pt")
    parser.add_argument('--ocr_weight', type=float, default=0.001)
    parser.add_argument('--ablation_weights', default=[0.001], type=float, nargs='+',
                        help="List of OCR weights to loop through")
    parser.add_argument("--scale", type=float, default=0.03, help="Percentage of dataset to use")
    parser.add_argument("--val_split", type=float, default=0)
    parser.add_argument("--batchSize", type=int, default=4)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--aug", type=int, default=0, help="1 to enable augmentation")
    parser.add_argument("--nEpochs", type=int, default=350)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--step", type=int, default=50)
    parser.add_argument("--cuda", action="store_true", default=True)
    parser.add_argument("--gpus", default="0", type=str)
    parser.add_argument("--resume", default="back_up_models/SR/ss-tsrn/ss-tsrn_300.pth", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--start-epoch", default=1, type=int)
    return parser.parse_args()


def get_model(arch_name, device):
    """Factory function to initialize models."""
    print(f"===> Building model: {arch_name.upper()}")
    arch_name = arch_name.lower()
    params = {}

    if arch_name == 'tsrn':
        params = {'scale_factor': 2, 'width': 128, 'height': 32,
                  'STN': True, 'srb_nums': 12, 'mask': True, 'hidden_units': 64}
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


def load_ocr_model(device):
    """Initializes and loads the frozen OCR model for loss calculation."""
    print("\n===> Building and loading OCR model...")

    # OCR Configuration
    class OCRConfig:
        Transformation = 'TPS'
        FeatureExtraction = 'ResNet'
        SequenceModeling = 'BiLSTM'
        Prediction = 'Attn'
        num_fiducial = 20
        input_channel = 1
        output_channel = 512
        hidden_size = 256
        num_class = 96
        imgH = 32
        imgW = 100
        batch_max_length = 35

    ocr_config = OCRConfig()
    netOCR = ModelOCR(ocr_config).to(device)

    # Load weights
    checkpoint_path = "back_up_models/OCR/TPS-ResNet-BiLSTM-Attn-case-sensitive.pth"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"OCR Checkpoint not found at {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    # Fix module keys if model was trained with DataParallel
    new_state_dict = OrderedDict((k[7:] if k.startswith('module.') else k, v) for k, v in state_dict.items())
    netOCR.load_state_dict(new_state_dict)

    # Freeze OCR model
    for p in netOCR.parameters():
        p.requires_grad = False
    netOCR.train()

    return netOCR


# =========================================================================
# 2. CORE TRAINING LOGIC
# =========================================================================

def train_one_epoch(opt, loader, model, ocr_proc, criterion, optimizer, epoch, device, ocr_weight):
    model.train()
    losses = {'img': [], 'ocr': []}

    optimizer.zero_grad()

    for i, batch in enumerate(loader, 1):
        input_img, real_img, ocr_label = batch
        input_img = input_img.to(device)
        real_img = real_img.to(device)

        # Handle channel differences based on architecture
        model_input = input_img if opt.arch == 'tsrn' else input_img[:, :3, :, :]

        # Forward Pass
        sr_image = model(model_input)
        sr_rgb = sr_image[:, :3, :, :] if sr_image.shape[1] == 4 else sr_image

        # Loss Calculation
        img_loss = criterion(real_img, sr_rgb)

        # OCR Loss (convert to grayscale for OCR)
        sr_gray = sr_rgb.mean(dim=1, keepdim=True)
        ocr_loss_val = ocr_proc.process(sr_gray, ocr_label)

        # Weighted Sum
        total_loss = (img_loss + (ocr_loss_val * ocr_weight)) / opt.accumulation
        total_loss.backward()

        losses['img'].append(img_loss.item())
        losses['ocr'].append(ocr_loss_val.item())

        if i % opt.accumulation == 0 or i == len(loader):
            optimizer.step()
            optimizer.zero_grad()

    avg_img = np.mean(losses['img'])
    avg_ocr = np.mean(losses['ocr'])
    print(f"Epoch {epoch} [Train] - Img: {avg_img:.4f}, OCR: {avg_ocr:.4f}")
    return {'img': avg_img, 'ocr': avg_ocr}


def validate_one_epoch(opt, loader, model, ocr_proc, criterion, epoch, device):
    model.eval()
    losses = {'img': [], 'ocr': []}

    if len(loader) == 0:
        return {'img': 0.0, 'ocr': 0.0}

    with torch.no_grad():
        for batch in loader:
            input_img, real_img, ocr_label = batch
            input_img = input_img.to(device)
            real_img = real_img.to(device)

            model_input = input_img if opt.arch == 'tsrn' else input_img[:, :3, :, :]
            sr_image = model(model_input)
            sr_rgb = sr_image[:, :3, :, :] if sr_image.shape[1] == 4 else sr_image

            img_loss = criterion(real_img, sr_rgb)
            sr_gray = sr_rgb.mean(dim=1, keepdim=True)
            ocr_loss_val = ocr_proc.process(sr_gray, ocr_label)

            losses['img'].append(img_loss.item())
            losses['ocr'].append(ocr_loss_val.item())

    avg_img = np.mean(losses['img'])
    avg_ocr = np.mean(losses['ocr'])
    print(f"Epoch {epoch} [Val]   - Img: {avg_img:.4f}, OCR: {avg_ocr:.4f}")
    return {'img': avg_img, 'ocr': avg_ocr}


# =========================================================================
# 3. DATA AUGMENTATION (GO BOTTOM)
# =========================================================================

class JointRandomAugment:
    """
    Applies the same random augmentation to a pair of low-res and high-res images.
    Safe for OCR-based training.
    """

    def __init__(self, max_rotation=5, max_translation=2, scale_range=(0.95, 1.05),
                 brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.9, 1.1),
                 gamma=(0.9, 1.1), swap_prob=0.5):
        self.max_rotation = max_rotation
        self.max_translation = max_translation
        self.scale_range = scale_range
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.gamma = gamma
        self.swap_prob = swap_prob

    def __call__(self, lr_img, hr_img):
        # 1. Geometric transforms
        angle = random.uniform(-self.max_rotation, self.max_rotation)
        tx = random.uniform(-self.max_translation, self.max_translation)
        ty = random.uniform(-self.max_translation, self.max_translation)
        scale = random.uniform(*self.scale_range)

        lr_aug = F.affine(lr_img, angle=angle, translate=(tx, ty), scale=scale, shear=0,
                          interpolation=F.InterpolationMode.BILINEAR)
        hr_aug = F.affine(hr_img, angle=angle, translate=(tx, ty), scale=scale, shear=0,
                          interpolation=F.InterpolationMode.BILINEAR)

        # 2. Photometric transforms
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

        # 3. Channel Shuffle (Optional)
        if random.random() < self.swap_prob:
            lr_rgb = lr_rgb[[2, 1, 0], :, :]
            hr_rgb = hr_rgb[[2, 1, 0], :, :]

        lr_aug[:3, :, :] = lr_rgb
        return lr_aug, hr_rgb


class AugmentedDataset(Dataset):
    """Wrapper to apply augmentation on the fly."""

    def __init__(self, original_dataset):
        self.original_dataset = original_dataset

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, index):
        input_img, real_img, ocr_label = self.original_dataset[index]
        augmenter = JointRandomAugment()
        aug_input, aug_real = augmenter(input_img, real_img)
        return aug_input, aug_real, ocr_label


# =========================================================================
# 4. SAVING & LOGGING FUNCTIONS (GO DOWN)
# =========================================================================

def save_run_summary(run_folder, opt, model, model_params, dataset_sizes, current_ocr_weight):
    summary_path = os.path.join(run_folder, 'run_summary.txt')
    total_params = sum(p.numel() for p in model.parameters())

    with open(summary_path, 'w') as f:
        f.write("=== Training Run Summary ===\n\n")
        f.write(f"Architecture:   {opt.arch.upper()}\n")
        f.write(f"OCR Weight:     {current_ocr_weight}\n")
        f.write(f"Learning Rate:  {opt.lr}\n")
        f.write(f"Batch Size:     {opt.batchSize}\n")
        f.write(f"Accumulation:   {opt.accumulation}\n\n")

        f.write("--- Dataset ---\n")
        f.write(f"Train Size:     {dataset_sizes['train']}\n")
        f.write(f"Val Size:       {dataset_sizes['val']}\n\n")

        f.write("--- Model Params ---\n")
        f.write(f"Total Params:   {total_params:,}\n")
        f.write(f"Hyperparams:    {model_params}\n")

    print(f"Run summary saved to {summary_path}")


def save_checkpoint(model, epoch, save_folder):
    checkpoint_dir = os.path.join(save_folder, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pth")

    state = {"epoch": epoch, "model_state_dict": model.state_dict()}
    torch.save(state, path)
    print(f"Checkpoint saved: {path}")


def save_metrics(metrics, save_folder):
    # Save CSV
    df = pd.DataFrame(metrics)
    df.to_csv(os.path.join(save_folder, 'metrics.csv'), index=False)

    # Save Plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    # Image Loss
    axs[0].plot(df['epoch'], df['train_img_loss'], 'o-', label='Train')
    axs[0].plot(df['epoch'], df['val_img_loss'], 'o-', label='Val')
    axs[0].set_title('Image Loss (MSE)')
    axs[0].set_xlabel('Epoch')
    axs[0].legend()

    # OCR Loss
    axs[1].plot(df['epoch'], df['train_ocr_loss'], 'o-', label='Train')
    axs[1].plot(df['epoch'], df['val_ocr_loss'], 'o-', label='Val')
    axs[1].set_title('OCR Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'loss_plot.png'))
    plt.close()


# =========================================================================
# 5. MAIN EXECUTION
# =========================================================================

def main():
    opt = parse_arguments()
    print("--- Training Configuration ---")
    print(opt)

    # --- Setup ---
    device = torch.device("cuda" if opt.cuda and torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        print(f"Using GPU: {opt.gpus}")
        torch.cuda.manual_seed(REPRODUCIBILITY_SEED)
    else:
        print("Using CPU")

    torch.manual_seed(REPRODUCIBILITY_SEED)
    cudnn.benchmark = True

    # --- Dataset ---
    print("\n===> Loading Dataset...")
    full_dataset = DatasetFromPT(opt.dataset)
    subset_size = int(len(full_dataset) * opt.scale)
    indices = list(range(len(full_dataset)))[:subset_size]

    study_subset = Subset(full_dataset, indices)
    val_size = int(opt.val_split * len(study_subset))
    train_size = len(study_subset) - val_size

    train_dataset = Subset(study_subset, range(train_size))
    val_dataset = Subset(study_subset, range(train_size, len(study_subset)))

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    if opt.aug > 0:
        print("Augmentation Enabled.")
        # AugmentedDataset class is defined below, but visible here
        train_dataset = AugmentedDataset(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True,
                              num_workers=opt.threads, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batchSize, shuffle=False,
                            num_workers=opt.threads, pin_memory=True)

    # --- OCR Setup ---
    netOCR = load_ocr_model(device)
    character = string.printable[:-6]
    converter = AttnLabelConverter(character)
    ocr_processor = OCRProcessor(netOCR, converter, device)
    mse_criterion = nn.L1Loss().to(device)

    # --- Loop over Weights (Ablation) ---
    weights_to_run = opt.ablation_weights if opt.ablation_weights else [opt.ocr_weight]

    start_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    parent_folder = os.path.join("experiments", f"{opt.arch}_{start_time_str}")
    os.makedirs(parent_folder, exist_ok=True)

    for ocr_weight in weights_to_run:
        run_name = f"weight_{ocr_weight}"
        print(f"\n{'=' * 30}\nRUNNING: {run_name}\n{'=' * 30}")

        run_folder = os.path.join(parent_folder, run_name)
        os.makedirs(run_folder, exist_ok=True)

        # Initialize SR Model
        torch.manual_seed(REPRODUCIBILITY_SEED)
        netSR, model_params = get_model(opt.arch, device)

        # Resume Logic
        start_epoch = opt.start_epoch
        if opt.resume and os.path.isfile(opt.resume):
            print(f"===> Resuming from: {opt.resume}")
            checkpoint = torch.load(opt.resume, map_location=device,weights_only=False)
            # Handle both saving formats (full dict vs state_dict only)
            netSR.load_state_dict(checkpoint['model'].state_dict())
            start_epoch = checkpoint.get('epoch', start_epoch) + 1


        save_run_summary(run_folder, opt, netSR, model_params,
                         {'train': len(train_dataset), 'val': len(val_dataset)}, ocr_weight)

        optimizer = torch.optim.AdamW(netSR.parameters(), lr=opt.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.step, gamma=0.5)

        metrics = {'epoch': [], 'train_img_loss': [], 'train_ocr_loss': [],
                   'val_img_loss': [], 'val_ocr_loss': []}

        for epoch in range(start_epoch, opt.nEpochs + 1):
            t_loss = train_one_epoch(opt, train_loader, netSR, ocr_processor, mse_criterion,
                                     optimizer, epoch, device, ocr_weight)
            v_loss = validate_one_epoch(opt, val_loader, netSR, ocr_processor, mse_criterion,
                                        epoch, device)

            scheduler.step()

            # Record Metrics
            metrics['epoch'].append(epoch)
            metrics['train_img_loss'].append(t_loss['img'])
            metrics['train_ocr_loss'].append(t_loss['ocr'])
            metrics['val_img_loss'].append(v_loss['img'])
            metrics['val_ocr_loss'].append(v_loss['ocr'])

            save_metrics(metrics, run_folder)

            if epoch % 50 == 0:
                save_checkpoint(netSR, epoch, run_folder)

        print(f"Finished Run: {run_name}")


if __name__ == "__main__":
    main()