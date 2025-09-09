import argparse
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from collections import OrderedDict
import os
import torch
import pandas as pd
import string
from datetime import datetime
import matplotlib.pyplot as plt

# --- Model Imports ---
from model.tsrn import TSRN
from Network.srresnet import _NetG as SRResNet
from model.rdn import RDN
from model.srcnn import SRCNN

# --- Helper Imports ---
from OCR.model import ModelOCR
from OCR.utils import AttnLabelConverter
from OCR.ocr_loss import OCRProcessor
from data_set_from_pt import DatasetFromPT  # Using the efficient .pt loader


class DynamicLossWeighter(nn.Module):
    """
    Learns to balance multiple loss terms using uncertainty weighting.
    L_total = 0.5 * exp(-log_var_1) * L_1 + 0.5 * exp(-log_var_2) * L_2 + log_var_1 + log_var_2
    """

    def __init__(self, num_losses=2):
        super().__init__()
        # We learn the log of the variance for numerical stability
        self.log_vars = nn.Parameter(torch.zeros(num_losses))

    def forward(self, img_loss, ocr_loss):
        # Image loss term
        precision_img = torch.exp(-self.log_vars[0])
        weighted_img_loss = precision_img * img_loss + self.log_vars[0]

        # OCR loss term
        precision_ocr = torch.exp(-self.log_vars[1])
        weighted_ocr_loss = precision_ocr * ocr_loss + self.log_vars[1]

        # We use a 0.5 factor for stability, common in implementations
        return 0.5 * (weighted_img_loss + weighted_ocr_loss)



def get_model(arch_name, device):
    """Initializes and returns the specified model architecture."""
    print(f"===> Building model: {arch_name.upper()}")
    arch_name = arch_name.lower()

    if arch_name == 'tsrn':
        model = TSRN(scale_factor=2, width=128, height=32, STN=True, srb_nums=12, mask=True, hidden_units=64)
    elif arch_name == 'srresnet':
        model = SRResNet(num_channels=64)
    elif arch_name == 'rdn':
        model = RDN(scale_factor=2)
    elif arch_name == 'srcnn':
        model = SRCNN(scale_factor=2)
    else:
        raise ValueError(f"Unknown architecture specified: {arch_name}")

    return model.to(device)


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Training Script with Dynamic Loss Weighting")

    # --- Core Arguments ---
    parser.add_argument('--arch', type=str, default="tsrn", choices=['tsrn', 'srresnet', 'rdn', 'srcnn'],
                        help='Model architecture to train.')
    parser.add_argument("--dataset", type=str, default="datasets/dataset.pt",
                        help="Path to the preprocessed .pt dataset file.")

    # --- Dataset and Dataloader Arguments ---
    parser.add_argument("--scale", type=float, default=0.05, help="Fraction of the dataset to use (e.g., 0.1 for 10%).")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Fraction of the data to use for validation (e.g., 0.1 for 10%).")
    parser.add_argument("--batchSize", type=int, default=2, help="Training batch size.")
    parser.add_argument("--accumulation", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader.")

    # --- Training Hyperparameters ---
    parser.add_argument("--nEpochs", type=int, default=114, help="Number of epochs to train for.")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate.")
    parser.add_argument("--step", type=int, default=500, help="Learning rate decay step.")

    # --- System and Checkpoint Arguments ---
    parser.add_argument("--cuda", action="store_false", help="Disable cuda training.")
    parser.add_argument("--gpus", default="0", type=str, help="GPU ids (e.g. 0 or 0,1).")
    parser.add_argument("--resume", default="", type=str, help="Path to checkpoint to resume from.")
    parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts).")

    opt = parser.parse_args()
    print("--- Training Script with Dynamic Loss Weighting ---")
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
    indices = torch.randperm(total_size).tolist()[:subset_size]
    study_subset = Subset(full_dataset, indices)
    print(f"Using a subset of {len(study_subset)} images ({opt.scale * 100:.1f}% of total).")

    val_size = int(opt.val_split * len(study_subset))
    train_size = len(study_subset) - val_size
    train_dataset, val_dataset = random_split(study_subset, [train_size, val_size])
    print(f"Split into {len(train_dataset)} training and {len(val_dataset)} validation images.")

    train_loader = DataLoader(dataset=train_dataset, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True,
                              drop_last=True, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=False,
                            pin_memory=True)

    # --- OCR Model Setup (Done Once) ---
    print("\n===> Building and loading OCR model...")
    from argparse import Namespace
    ocr_config = Namespace(Transformation='TPS', FeatureExtraction='ResNet', SequenceModeling='BiLSTM',
                           Prediction='Attn',
                           num_fiducial=20, input_channel=1, output_channel=512, hidden_size=256,
                           num_class=96, imgH=32, imgW=100, batch_max_length=35, device=device)
    netOCR = ModelOCR(ocr_config).to(device)

    state_dict = torch.load("back_up_models/OCR/TPS-ResNet-BiLSTM-Attn-case-sensitive.pth", map_location=device,
                            weights_only=True)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
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

    # --- Main Training Section ---
    start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_folder = os.path.join("experiments", f"{opt.arch}_{start_time}_dynamic_weights")
    os.makedirs(run_folder, exist_ok=True)
    print(f"Results will be saved in: {run_folder}")

    # Initialize model
    torch.manual_seed(reproducibility_seed)
    netSR = get_model(opt.arch, device)

    # Initialize the dynamic loss weighter
    loss_weighter = DynamicLossWeighter(num_losses=2).to(device)
    # loss_weighter = ConstrainedDynamicLossWeighter().to(device)

    # ######################################################################
    # ############# شروع تغییرات: تعریف نرخ یادگیری متفاوت #############
    # ######################################################################

    # نرخ یادگیری برای وزن‌ها ۱۰۰ برابر نرخ یادگیری اصلی تعریف می‌شود
    lr_weighter = opt.lr * 10

    print(f"===> Using separate learning rates: Model LR = {opt.lr}, Weighter LR = {lr_weighter}")

    optimizer = torch.optim.AdamW([
        # گروه اول: پارامترهای مدل اصلی با نرخ یادگیری پیش‌فرض
        {'params': netSR.parameters()},

        # گروه دوم: پارامترهای مربوط به وزن‌دهی با نرخ یادگیری مشخص و متفاوت
        {'params': loss_weighter.parameters(), 'lr': lr_weighter}

    ], lr=opt.lr, weight_decay=1e-5)  # lr=opt.lr به عنوان نرخ پیش‌فرض برای گروه‌هایی عمل می‌کند که lr ندارند

    # ######################################################################
    # ############## پایان تغییرات: تعریف نرخ یادگیری متفاوت ###############
    # ######################################################################

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.step, gamma=0.1)

    metrics = {
        'epoch': [], 'train_img_loss': [], 'train_ocr_loss': [],
        'val_img_loss': [], 'val_ocr_loss': [],
        'img_weight': [], 'ocr_weight': []
    }

    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        train_losses = train_one_epoch(opt, train_loader, netSR, loss_weighter, ocr_processor, mse_criterion, optimizer,
                                       epoch, device, opt.accumulation)
        val_losses = validate_one_epoch(opt, val_loader, netSR, ocr_processor, mse_criterion, epoch, device)
        scheduler.step()

        # Get the learned weights for logging
        # The effective weight is exp(-log_var)
        learned_weights = torch.exp(-loss_weighter.log_vars).detach().cpu().numpy()
        img_weight, ocr_weight = learned_weights[0], learned_weights[1]

        print(f"--- Epoch {epoch} Learned Weights --- Img Weight: {img_weight:.4f}, OCR Weight: {ocr_weight:.4f}")

        # Log metrics
        metrics['epoch'].append(epoch)
        metrics['train_img_loss'].append(train_losses['img'])
        metrics['train_ocr_loss'].append(train_losses['ocr'])
        metrics['val_img_loss'].append(val_losses['img'])
        metrics['val_ocr_loss'].append(val_losses['ocr'])
        metrics['img_weight'].append(img_weight)
        metrics['ocr_weight'].append(ocr_weight)

        save_and_plot_metrics(metrics, run_folder)

        if epoch % 50 == 0:
            save_checkpoint(netSR, epoch, run_folder)

    print(f"{'=' * 25} FINISHED DYNAMIC TRAINING {'=' * 25}")


def train_one_epoch(opt, data_loader, netSR, loss_weighter, ocr_processor, mse_criterion, optimizer, epoch, device,
                    accumulation_steps=1):
    netSR.train()
    loss_weighter.train()  # Ensure the weighter is in training mode
    img_loss_list, ocr_loss_list = [], []

    optimizer.zero_grad()
    for iteration, batch in enumerate(data_loader, 1):
        input_img, real_img, ocr_label = batch
        input_img, real_img = input_img.to(device), real_img.to(device)

        if opt.arch == 'tsrn':
            model_input = input_img
        else:
            model_input = input_img[:, :3, :, :]

        sr_image = netSR(model_input)
        sr_image_rgb = sr_image[:, :3, :, :] if sr_image.shape[1] == 4 else sr_image

        # Calculate raw losses
        img_loss = mse_criterion(real_img, sr_image_rgb)
        sr_image_gray = sr_image_rgb.mean(dim=1, keepdim=True)
        ocr_loss = ocr_processor.process(sr_image_gray, ocr_label)

        # Append raw losses for logging purposes
        img_loss_list.append(img_loss.item())
        ocr_loss_list.append(ocr_loss.item())

        # Get the dynamically weighted total loss
        total_loss = loss_weighter(img_loss, ocr_loss) / accumulation_steps
        total_loss.backward()

        if (iteration % accumulation_steps == 0) or (iteration == len(data_loader)):
            torch.nn.utils.clip_grad_norm_(netSR.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

    avg_losses = {'img': np.mean(img_loss_list), 'ocr': np.mean(ocr_loss_list)}
    print(
        f"--- Epoch {epoch} Training Summary --- Img Loss: {avg_losses['img']:.4f}, OCR Loss: {avg_losses['ocr']:.4f}")
    return avg_losses


def validate_one_epoch(opt, data_loader, netSR, ocr_processor, mse_criterion, epoch, device):
    netSR.eval()
    img_loss_list, ocr_loss_list = [], []

    with torch.no_grad():
        for batch in data_loader:
            input_img, real_img, ocr_label = batch
            input_img, real_img = input_img.to(device), real_img.to(device)

            if opt.arch == 'tsrn':
                model_input = input_img
            else:
                model_input = input_img[:, :3, :, :]

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
    plot_path = os.path.join(save_folder, 'metrics_plot.png')
    df = pd.DataFrame(metrics)
    df.to_csv(csv_path, index=False)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axs = plt.subplots(1, 3, figsize=(22, 6))  # Changed to 3 subplots

    # Plot 1: Image Reconstruction Loss
    axs[0].plot(df['epoch'], df['train_img_loss'], 'o-', label='Train Image Loss')
    axs[0].plot(df['epoch'], df['val_img_loss'], 'o-', label='Validation Image Loss')
    axs[0].set_title('Image Reconstruction Loss (MSE)')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    # Plot 2: OCR Attention Loss
    axs[1].plot(df['epoch'], df['train_ocr_loss'], 'o-', label='Train OCR Loss')
    axs[1].plot(df['epoch'], df['val_ocr_loss'], 'o-', label='Validation OCR Loss')
    axs[1].set_title('OCR Attention Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].legend()

    # Plot 3: Dynamically Learned Weights
    axs[2].plot(df['epoch'], df['img_weight'], 's-', label='Image Loss Weight')
    axs[2].plot(df['epoch'], df['ocr_weight'], 's-', label='OCR Loss Weight')
    axs[2].set_title('Dynamically Learned Loss Weights')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('Weight Value (exp(-log_var))')
    axs[2].legend()
    # Use a log scale if weights vary by orders of magnitude
    # axs[2].set_yscale('log')

    fig.suptitle(f"Metrics for {os.path.basename(save_folder)}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(plot_path)
    plt.close()


if __name__ == "__main__":
    main()