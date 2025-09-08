import argparse
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from model.tsrn import TSRN
from torch.utils.data import DataLoader, random_split
from OCR.model import ModelOCR
from OCR.utils import AttnLabelConverter
from OCR.ocr_loss import OCRProcessor
from dataset_prepration_mask import DatasetFromImages
import random
import os
import torch
import pandas as pd
import string
import time
from datetime import datetime
import matplotlib.pyplot as plt
import cv2

# --- Argument Parser Setup ---
parser = argparse.ArgumentParser(description="PyTorch TSRN Ablation Study")
parser.add_argument("--batchSize", type=int, default=32, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=250, help="number of epochs to train for")
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate, default=0.0001")
parser.add_argument("--step", type=int, default=1000, help="Learning rate decay step")
parser.add_argument("--cuda", action="store_false", help="Use cuda? Disables CUDA if present.")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=4, help="Number of threads for data loader")
parser.add_argument("--pretrained", default="", type=str, help="Path to pretrained model (default: none)")
parser.add_argument("--gpus", default="0", type=str, help="GPU ids (default: 0)")


# --- Core Ablation Study Logic ---
def run_ablation_study():
    """
    Main function to set up and run the ablation study.
    """
    opt = parser.parse_args()
    print("--- Starting Ablation Study ---")
    print(opt)

    # Define the weights to be tested in the ablation study
    ablation_weights = [1, 0.5, 0.1, 0.01, 0.005, 0.001]

    # --- Device Configuration ---
    device = torch.device("cuda" if opt.cuda and torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using GPU: {opt.gpus}")
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    else:
        print("Using CPU.")

    # --- Reproducibility ---
    # Set a fixed seed for dataset splitting to ensure fairness
    reproducibility_seed = 42
    torch.manual_seed(reproducibility_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(reproducibility_seed)
    cudnn.benchmark = True

    # --- Dataset Loading and Splitting (Done ONCE) ---
    print("\n===> Loading and splitting dataset...")
    sr_data_path = "datasets/data"

    # 1. Choose a subset of the total dataset (e.g., 2% of images)
    full_subset = DatasetFromImages(sr_data_path, scale=0.025)
    print(f"Loaded a subset with {len(full_subset)} images.")

    # 2. Split this subset into training (90%) and validation (10%)
    val_size = int(0.1 * len(full_subset))
    train_size = len(full_subset) - val_size
    train_dataset, val_dataset = random_split(full_subset, [train_size, val_size])

    print(f"Split into {len(train_dataset)} training images and {len(val_dataset)} validation images.")

    # Create DataLoaders from the splits
    train_loader = DataLoader(
        dataset=train_dataset, num_workers=opt.threads, batch_size=opt.batchSize,
        shuffle=True, drop_last=True, pin_memory=device.type == "cuda"
    )
    val_loader = DataLoader(
        dataset=val_dataset, num_workers=opt.threads, batch_size=opt.batchSize,
        shuffle=False, drop_last=False, pin_memory=device.type == "cuda"
    )

    # --- OCR Model Setup (Done ONCE) ---
    print("\n===> Building and loading OCR model...")
    from argparse import Namespace
    ocr_config = Namespace(
        Transformation='TPS', FeatureExtraction='ResNet', SequenceModeling='BiLSTM', Prediction='Attn',
        num_fiducial=20, input_channel=1, output_channel=512, hidden_size=256,
        num_class=96, imgH=32, imgW=100, batch_max_length=35
    )
    netOCR = ModelOCR(ocr_config)
    netOCR = torch.nn.DataParallel(netOCR).to(device)
    netOCR.load_state_dict(
        torch.load("back_up_models/OCR/TPS-ResNet-BiLSTM-Attn-case-sensitive.pth", map_location=device))
    netOCR.eval()  # OCR model is only for inference

    character = string.printable[:-6]
    converter = AttnLabelConverter(character)
    ocr_processor = OCRProcessor(netOCR, converter, opt.cuda)
    mse_criterion = nn.MSELoss(reduction='mean').to(device)

    # --- Main Ablation Loop ---
    start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    parent_folder = os.path.join("ablation_results", start_time)
    os.makedirs(parent_folder, exist_ok=True)
    print(f"\nResults will be saved in: {parent_folder}")

    for ocr_weight in ablation_weights:
        print(f"\n{'=' * 20} STARTING RUN FOR OCR WEIGHT: {ocr_weight} {'=' * 20}")

        # Create a specific directory for this run's results
        run_folder = os.path.join(parent_folder, f"weight_{ocr_weight}")
        os.makedirs(run_folder, exist_ok=True)

        # --- Re-initialize TSRN Model and Optimizer for a fair trial ---
        torch.manual_seed(reproducibility_seed)  # Ensure model weights start the same
        netSR = TSRN(
            scale_factor=2, width=128, height=32, STN=True,
            srb_nums=3, mask=True, hidden_units=32
        ).to(device)

        optimizerG = torch.optim.AdamW(netSR.parameters(), lr=opt.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizerG, step_size=opt.step, gamma=0.1)

        metrics = {'epoch': [], 'train_img_loss': [], 'train_ocr_loss': [], 'val_img_loss': [], 'val_ocr_loss': []}

        # --- Training & Validation Loop for the current weight ---
        for epoch in range(opt.start_epoch, opt.nEpochs + 1):
            train_losses = train_one_epoch(train_loader, netSR, ocr_processor, mse_criterion, optimizerG, epoch, opt,
                                           device, ocr_weight)
            val_losses = validate_one_epoch(val_loader, netSR, ocr_processor, mse_criterion, epoch, opt, device,
                                            ocr_weight)

            scheduler.step()

            # Log metrics
            metrics['epoch'].append(epoch)
            metrics['train_img_loss'].append(train_losses['img'])
            metrics['train_ocr_loss'].append(train_losses['ocr'])
            metrics['val_img_loss'].append(val_losses['img'])
            metrics['val_ocr_loss'].append(val_losses['ocr'])

            save_and_plot_metrics(metrics, run_folder)

            if epoch % 10 == 0:  # Save checkpoint every 10 epochs
                save_checkpoint(netSR, epoch, run_folder)

        print(f"{'=' * 20} FINISHED RUN FOR OCR WEIGHT: {ocr_weight} {'=' * 20}")


def train_one_epoch(data_loader, netSR, ocr_processor, mse_criterion, optimizerG, epoch, opt, device, ocr_weight,
                    accumulation_steps=4):
    netSR.train()
    img_loss_list, ocr_loss_list = [], []
    initial_img_weight = 1.0  # Constant image weight

    optimizerG.zero_grad()
    for iteration, batch in enumerate(data_loader, 1):
        input_img, real_img, ocr_label = batch
        input_img, real_img = input_img.to(device), real_img.to(device)

        sr_image = netSR(input_img)

        # Handle 4-channel output (if mask is present)
        sr_image_rgb = sr_image[:, :3, :, :] if sr_image.shape[1] == 4 else sr_image

        # Image Reconstruction Loss
        img_loss = mse_criterion(real_img, sr_image_rgb)

        # OCR Loss
        sr_image_gray = sr_image_rgb.mean(dim=1, keepdim=True)
        ocr_loss = ocr_processor.process(sr_image_gray, ocr_label)

        # Total Loss with current OCR weight
        total_loss = (img_loss * initial_img_weight + ocr_loss * ocr_weight) / accumulation_steps
        total_loss.backward()

        img_loss_list.append(img_loss.item())
        ocr_loss_list.append(ocr_loss.item())

        if iteration % accumulation_steps == 0 or iteration == len(data_loader):
            optimizerG.step()
            optimizerG.zero_grad()

        if iteration % 50 == 0:
            print(f"Epoch [{epoch}/{opt.nEpochs}], Iter [{iteration}/{len(data_loader)}], "
                  f"Img Loss: {img_loss.item():.4f}, OCR Loss: {ocr_loss.item():.4f}")

    avg_losses = {'img': np.mean(img_loss_list), 'ocr': np.mean(ocr_loss_list)}
    print(
        f"--- Epoch {epoch} Training Summary --- Img Loss: {avg_losses['img']:.4f}, OCR Loss: {avg_losses['ocr']:.4f}")
    return avg_losses


def validate_one_epoch(data_loader, netSR, ocr_processor, mse_criterion, epoch, opt, device, ocr_weight):
    netSR.eval()
    img_loss_list, ocr_loss_list = [], []
    initial_img_weight = 1.0

    with torch.no_grad():
        for batch in data_loader:
            input_img, real_img, ocr_label = batch
            input_img, real_img = input_img.to(device), real_img.to(device)

            sr_image = netSR(input_img)
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
    state = {"epoch": epoch, "model": model.state_dict()}
    torch.save(state, model_out_path)
    print(f"Checkpoint saved to {model_out_path}")


def save_and_plot_metrics(metrics, save_folder):
    csv_path = os.path.join(save_folder, 'metrics.csv')
    plot_path = os.path.join(save_folder, 'loss_plot.png')

    # Save metrics to CSV
    df = pd.DataFrame(metrics)
    df.to_csv(csv_path, index=False)

    # Plot metrics
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    # Plot Image Loss (Train vs Val)
    axs[0].plot(df['epoch'], df['train_img_loss'], 'o-', label='Train Image Loss')
    axs[0].plot(df['epoch'], df['val_img_loss'], 'o-', label='Validation Image Loss')
    axs[0].set_title('Image Reconstruction Loss (MSE)')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    # Plot OCR Loss (Train vs Val)
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
    run_ablation_study()