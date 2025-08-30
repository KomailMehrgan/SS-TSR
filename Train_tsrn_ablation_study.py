import argparse
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from model.tsrn import TSRN
from Network.content_network import content_model
import cv2
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
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
from argparse import Namespace

# Training settings
parser = argparse.ArgumentParser(description="PyTorch TSRN Ablation Study")
parser.add_argument("--batchSize", type=int, default=16, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=5, help="number of epochs to train for each beta value")
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate, default=0.0001")
parser.add_argument("--step", type=int, default=1000, help="Learning rate decay step, Default: 75")
parser.add_argument("--cuda", action="store_false", help="Use cuda?")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--gpus", default="0", type=str, help="GPU ids (default: 0)")


def main():
    global opt, netContent
    opt = parser.parse_args()
    print("--- Starting Ablation Study ---")

    # === ABLATION STUDY SETUP ===
    # Define the five beta values (OCR loss weights) you want to test.
    # This range tests different orders of magnitude around your original value of 0.002.
    beta_values = [0.0005, 0.001, 0.002, 0.005, 0.01]

    # Create a main directory for all ablation study results
    study_start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    study_parent_folder = os.path.join("back_up_models/SR/RES", f"ablation_study_{study_start_time}")
    os.makedirs(study_parent_folder, exist_ok=True)
    print(f"Results for all runs will be saved in: {study_parent_folder}\n")

    device = torch.device("cuda" if opt.cuda and torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using GPU: {opt.gpus}")
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    else:
        print("Using CPU.")

    # === LOOP FOR EACH BETA VALUE ===
    for beta in beta_values:
        print(f"\n{'=' * 20} Starting Run for Beta = {beta} {'=' * 20}")

        # Create a unique subfolder for this specific run's results
        run_folder_name = f"run_beta_{beta}"
        current_run_parent_folder = os.path.join(study_parent_folder, run_folder_name)
        os.makedirs(current_run_parent_folder, exist_ok=True)
        print(f"Saving checkpoints and metrics for this run in: {current_run_parent_folder}")

        # Set a consistent seed for reproducibility across runs
        opt.seed = 42  # Use a fixed seed for fair comparison
        print("Random Seed: ", opt.seed)
        torch.manual_seed(opt.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(opt.seed)

        cudnn.benchmark = True

        print("===> Loading datasets")
        sr_data_path = "datasets/data"
        train_set = DatasetFromImages(sr_data_path, scale=0.01)
        training_data_loader = DataLoader(
            dataset=train_set,
            num_workers=opt.threads,
            batch_size=opt.batchSize,
            shuffle=True,
            drop_last=True,
            pin_memory=True if device.type == "cuda" else False,
            persistent_workers=True
        )

        # === CRITICAL STEP: Re-initialize model and optimizer for each run ===
        # This ensures each ablation run is independent and starts from scratch.
        print("===> Re-initializing TSRN model and optimizer")
        netSR = TSRN(
            scale_factor=2, width=128, height=32, STN=True,
            srb_nums=5, mask=True, hidden_units=64
        ).to(device)

        optimizerG = torch.optim.AdamW(netSR.parameters(), lr=opt.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizerG, step_size=opt.step, gamma=0.1)

        print("===> Loading OCR model")
        config = Namespace(
            Transformation='TPS', FeatureExtraction='ResNet', SequenceModeling='BiLSTM',
            Prediction='Attn', num_fiducial=20, input_channel=1, output_channel=512,
            hidden_size=256, num_class=96, imgH=32, imgW=100, batch_max_length=35
        )
        netOCR = ModelOCR(config)
        netOCR = torch.nn.DataParallel(netOCR).to(device)
        netOCR.load_state_dict(
            torch.load("back_up_models/OCR/TPS-ResNet-BiLSTM-Attn-case-sensitive.pth", map_location=device))

        character = string.printable[:-6]
        converter = AttnLabelConverter(character)

        print("===> Building criterions")
        mse_criterion = nn.MSELoss(reduction='mean').to(device)

        # Initialize metrics for this run
        metrics = {'img_loss': [], 'ocr_loss': []}

        print("===> Starting Training Loop for this run")
        for epoch in range(1, opt.nEpochs + 1):
            train(training_data_loader, optimizerG, netSR, netOCR, mse_criterion, epoch, metrics, converter,
                  current_run_parent_folder, ocr_weight=beta)
            save_checkpoint(netSR, epoch, current_run_parent_folder)

        print(f"--- Finished Run for Beta = {beta} ---")

    print("\n{'='*20} Ablation Study Complete! {'='*20}")


def train(data_loader, optimizerG, netSR, netOCR, mse_criterion, epoch, metrics, converter, parent_folder, ocr_weight,
          accumulation_steps=4):
    ocr_processor = OCRProcessor(netOCR, converter, opt.cuda)
    netSR.train()
    netOCR.eval()

    # Image loss weight is fixed at 1.0
    img_weight = 1.0

    img_loss_list, ocr_loss_list = [], []
    optimizerG.zero_grad()

    for iteration, batch in enumerate(data_loader, 1):
        input_data, realImage, ocr_label = batch
        if opt.cuda:
            input_data, realImage = input_data.to("cuda"), realImage.to("cuda")

        SrImage = netSR(input_data)
        SrImage_img = SrImage[:, :3, :, :] if SrImage.shape[1] == 4 else SrImage

        img_loss = mse_criterion(realImage, SrImage_img)

        SrImage_img_ocr = SrImage_img.mean(dim=1, keepdim=True) if SrImage_img.shape[1] == 3 else SrImage_img
        loss_ocr = ocr_processor.process(SrImage_img_ocr, ocr_label)

        # === DYNAMIC BETA USED HERE ===
        # The ocr_weight (beta) is now passed as an argument to the function.
        total_loss = (img_loss * img_weight + loss_ocr * ocr_weight) / accumulation_steps
        total_loss.backward()

        img_loss_list.append(img_loss.item())
        ocr_loss_list.append(loss_ocr.item())

        if (iteration + 1) % accumulation_steps == 0:
            optimizerG.step()
            optimizerG.zero_grad()

        if (iteration + 1) % 10 == 0:  # Print less frequently to avoid clutter
            print(f"Epoch [{epoch}], Iter [{iteration + 1}/{len(data_loader)}], "
                  f"Img Loss: {img_loss.item():.4f}, OCR Loss: {loss_ocr.item():.4f}, "
                  f"Total Loss (scaled): {total_loss.item():.4f}")

    if (iteration + 1) % accumulation_steps != 0:
        optimizerG.step()
        optimizerG.zero_grad()

    metrics['img_loss'].append(np.mean(img_loss_list))
    metrics['ocr_loss'].append(np.mean(ocr_loss_list))
    print_metrics_table(metrics, epoch, parent_folder)


def save_checkpoint(model, epoch, parent_folder):
    checkpoint_dir = os.path.join(parent_folder, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_out_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pth")
    state = {"epoch": epoch, "model_state_dict": model.state_dict()}
    torch.save(state, model_out_path)
    print(f"Checkpoint saved to {model_out_path}")


def print_metrics_table(metrics, epoch, parent_folder):
    save_path = os.path.join(parent_folder, 'training_metrics.csv')
    plot_save_path = os.path.join(parent_folder, 'training_plot.png')

    metrics_df = pd.DataFrame(metrics)

    # Save the full history to CSV
    metrics_df.to_csv(save_path, index_label='Epoch')

    print(f"\nMetrics at the end of Epoch {epoch} for this run:\n")
    print(metrics_df.tail(1))  # Print only the latest epoch's metrics

    # Generate and save the plot
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    axs[0].plot(metrics_df['img_loss'], label='Image Loss', marker='o')
    axs[0].set_title('Image Loss over Epochs')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(metrics_df['ocr_loss'], label='OCR Loss', marker='o', color='orange')
    axs[1].set_title('OCR Loss over Epochs')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(plot_save_path)
    plt.close()


if __name__ == "__main__":
    main()