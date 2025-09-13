import argparse
import numpy as np
import pandas as pd
import string
import os
from datetime import datetime
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, random_split, Subset
from torch.amp import autocast, GradScaler
from collections import OrderedDict

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


def get_model(arch_name, device):
    """Initializes and returns the specified model architecture."""
    arch_name = arch_name.lower()
    print(f"===> Building model: {arch_name.upper()}")

    if arch_name == 'tsrn':
        model = TSRN(scale_factor=2, width=128, height=32, STN=True, srb_nums=12, mask=True, hidden_units=64)
    elif arch_name == 'srresnet':
        model = SRResNet(num_channels=64)
    elif arch_name == 'rdn':
        model = RDN(scale_factor=2)
    elif arch_name == 'srcnn':
        model = SRCNN(scale_factor=2)
    else:
        raise ValueError(f"Unknown architecture: {arch_name}")
    return model.to(device)


def main():
    """Main function to parse arguments and orchestrate the training process."""
    parser = argparse.ArgumentParser(
        description="Optimized SR Training Script with OCR supervision and Gradient Accumulation")

    # --- Core Arguments ---
    parser.add_argument('--arch', type=str, default="tsrn", choices=['tsrn', 'srresnet', 'rdn', 'srcnn'],
                        help='Model architecture.')
    parser.add_argument("--dataset", type=str, default="datasets/dataset.pt",
                        help="Path to the preprocessed .pt dataset file.")

    # --- Training Mode Arguments ---
    parser.add_argument('--ocr_weight', type=float, default=0.01, help='Weight for the OCR loss.')
    parser.add_argument('--ablation_weights', default=[0,0.001,0.01,0.1,1], type=float, nargs='+',
                        help='List of OCR weights for ablation study.')

    # --- Dataset and Dataloader Arguments ---
    parser.add_argument("--scale", type=float, default=1, help="Fraction of the dataset to use.")
    parser.add_argument("--val_split", type=float, default=0.1, help="Fraction of data for validation.")
    parser.add_argument("--batchSize", type=int, default=128, help="Training batch size.")
    parser.add_argument("--accumulation_steps", type=int, default=1,
                        help="Number of steps to accumulate gradients before updating weights.")
    parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader.")

    # --- Training Hyperparameters ---
    parser.add_argument("--nEpochs", type=int, default=200, help="Number of epochs to train for.")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate.")
    parser.add_argument("--step", type=int, default=1000, help="Learning rate decay step.")

    # --- System and Checkpoint Arguments ---
    parser.add_argument("--cuda", action="store_false", help="Disable cuda training.")
    parser.add_argument("--gpus", default="0", type=str, help="GPU ids (e.g. 0 or 0,1).")
    parser.add_argument("--resume", default="", type=str, help="Path to checkpoint to resume from.")
    parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts).")

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
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    netOCR.load_state_dict(new_state_dict)
    # 1. Freeze OCR parameters as before
    for p in netOCR.parameters():
        p.requires_grad = False

    # 2. Set the entire model to TRAIN mode (to satisfy the RNN)
    netOCR.train()

    # 3. CRITICAL: Manually set all BatchNorm layers to EVAL mode for stability
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
        netSR = get_model(opt.arch, device)
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

            # Log metrics
            metrics['epoch'].append(epoch)
            metrics['train_img_loss'].append(train_losses['img'])
            metrics['train_ocr_loss'].append(train_losses['ocr'])
            metrics['val_img_loss'].append(val_losses['img'])
            metrics['val_ocr_loss'].append(val_losses['ocr'])
            save_and_plot_metrics(metrics, run_folder)

            # print(f"\n===== Epoch {epoch} Summary =====")
            # print(f"Train Image Loss: {train_losses['img']:.6f}")
            # print(f"Train OCR Loss  : {train_losses['ocr']:.6f}")
            # print(f"Val Image Loss  : {val_losses['img']:.6f}")
            # print(f"Val OCR Loss    : {val_losses['ocr']:.6f}")
            # print("===============================\n")

            if epoch % 50 == 0:
                save_checkpoint(netSR, epoch, run_folder)


def train_one_epoch(opt, data_loader, netSR, ocr_processor, mse_criterion, optimizer, epoch, device, ocr_weight,
                    scaler):
    """Trains the model for one epoch with gradient accumulation."""
    netSR.train()
    img_losses, ocr_losses = [], []

    # Zero gradients at the beginning of the epoch for the first accumulation cycle.
    optimizer.zero_grad()

    for iter_idx, batch in enumerate(data_loader, 1):
        input_img, real_img, ocr_label = batch
        input_img, real_img = input_img.to(device), real_img.to(device)

        with autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            # Forward pass
            model_input = input_img if opt.arch == 'tsrn' else input_img[:, :3, :, :]
            sr_image = netSR(model_input)
            sr_image_rgb = sr_image[:, :3, :, :] if sr_image.shape[1] >= 3 else sr_image
            img_loss = mse_criterion(real_img, sr_image_rgb)
            sr_gray = sr_image_rgb.mean(dim=1, keepdim=True)
            ocr_loss = ocr_processor.process(sr_gray, ocr_label)
            total_loss = img_loss + ocr_weight * ocr_loss

            # Normalize loss to average over accumulation steps.
            if opt.accumulation_steps > 1:
                total_loss = total_loss / opt.accumulation_steps

        # Scale loss and call backward() to accumulate scaled gradients.
        scaler.scale(total_loss).backward()

        # Update weights only after accumulating for the specified number of steps.
        if (iter_idx % opt.accumulation_steps == 0) or (iter_idx == len(data_loader)):
            # --- START OF FIX ---
            scaler.unscale_(optimizer)  # Unscales the gradients in-place
            torch.nn.utils.clip_grad_norm_(netSR.parameters(), max_norm=1.0)  # Now clip the correct gradients
            # --- END OF FIX ---
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()  # Zero gradients for the next accumulation cycle.

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
    # Correctly unwrap the model if it was compiled with torch.compile
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