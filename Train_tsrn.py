import argparse
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from model.tsrn import TSRN  # Replacing SRResNet with TSRN
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

# Training settings
parser = argparse.ArgumentParser(description="PyTorch TSRN")

parser.add_argument("--batchSize", type=int, default=64, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=5, help="number of epochs to train for")
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate, default=0.0001")
parser.add_argument("--step", type=int, default=1000, help="Learning rate decay step, Default: 75")
parser.add_argument("--cuda", action="store_false", help="Use cuda?")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--pretrained", default="", type=str, help="Path to pretrained model (default: none)")
parser.add_argument("--clamp", type=float, default=5)
parser.add_argument("--gpus", default="0", type=str, help="GPU ids (default: 0)")


def main():
    global opt, netContent
    opt = parser.parse_args()
    print(opt)

    start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    parent_folder = os.path.join("back_up_models/SR/RES", start_time,"_TSRN")
    os.makedirs(parent_folder, exist_ok=True)

    device = torch.device("cuda" if opt.cuda and torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using GPU: {opt.gpus}")
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    else:
        print("Using CPU.")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print("===> Loading datasets")
    sr_data_path = "datasets/data"
    train_set = DatasetFromImages(sr_data_path, scale=1)
    training_data_loader = DataLoader(
        dataset=train_set,
        num_workers=opt.threads,
        batch_size=opt.batchSize,
        shuffle=True,
        drop_last=True,
        pin_memory=True if device.type == "cuda" else False,
        persistent_workers=True
    )

    print("===> Building TSRN model")
    netSR = TSRN(
        scale_factor=2,
        width=128,
        height=32,
        STN=True,
        srb_nums=12,
        mask=True,
        hidden_units=64
    )

    print("===> Loading OCR model")
    from argparse import Namespace
    config = Namespace(
        Transformation='TPS',
        FeatureExtraction='ResNet',
        SequenceModeling='BiLSTM',
        Prediction='Attn',
        num_fiducial=20,
        input_channel=1,
        output_channel=512,
        hidden_size=256,
        num_class=96,  # Number of characters (alphabet + EOS + PAD)
        imgH=32,
        imgW=100,
        batch_max_length=35
    )
    netOCR = ModelOCR(config)
    netOCR = torch.nn.DataParallel(netOCR).to(device)
    netOCR.load_state_dict(torch.load("back_up_models/OCR/TPS-ResNet-BiLSTM-Attn-case-sensitive.pth", map_location=device))


    # case+farsi 106
    # farsi = "۰۱۲۳۴۵۶۷۸۹"
    # character = string.printable[:-6]   + farsi


    #case 96
    character = string.printable[:-6]

    #Normal 36
    # character = '0123456789abcdefghijklmnopqrstuvwxyz'



    converter = AttnLabelConverter(character)

    print("===> Building criterions")
    mse_criterion = nn.MSELoss(reduction='mean')

    if device.type == "cuda":
        netSR = netSR.to(device)
        mse_criterion = mse_criterion.to(device)

    if opt.resume or opt.pretrained:
        weights_path = opt.resume or opt.pretrained
        if os.path.isfile(weights_path):
            print(f"=> loading TSRN model from '{weights_path}'")
            checkpoint = torch.load(weights_path, map_location=device,weights_only=False)
            # netSR.load_state_dict(checkpoint['state_dict_G'])
            model_object = checkpoint['model']
            state_dict = model_object.state_dict()
            netSR.load_state_dict(state_dict)

        else:
            print(f"=> no model found at '{weights_path}'")

    print("===> Setting Optimizer")
    optimizerG = torch.optim.AdamW(netSR.parameters(), lr=opt.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizerG, step_size=opt.step, gamma=0.1)

    metrics = {'img_loss': [], 'ocr_loss': []}

    print("===> Training")

    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        if epoch % 1500 == 0:
            print(f"Completed epoch {epoch}. Pausing for 3 minutes.")
            time.sleep(250)
            print("Resuming training...")

        train(training_data_loader, optimizerG, netSR, netOCR, mse_criterion, epoch, metrics, converter, parent_folder)
        save_checkpoint(netSR, epoch, parent_folder)

def train(data_loader, optimizerG, netSR, netOCR, mse_criterion, epoch, metrics, converter, parent_folder, accumulation_steps=4):
    ocr_processor = OCRProcessor(netOCR, converter, opt.cuda)

    netSR.train()
    netOCR.eval()


    initial_img_weight = 1
    initial_ocr_weight = 0.002

    img_loss_list, mean_loss_ocr_list = [], []

    optimizerG.zero_grad()  # initialize once before loop

    for iteration, batch in enumerate(data_loader, 1):
        input, realImage, ocr_label = batch
        if opt.cuda:
            input, realImage = input.to("cuda"), realImage.to("cuda")

        SrImage = netSR(input)

        # If output has 4 channels, remove the mask channel before loss computation
        if SrImage.shape[1] == 4:
            SrImage_img = SrImage[:, :3, :, :]  # keep only RGB
        else:
            SrImage_img = SrImage

        img_loss = mse_criterion(realImage, SrImage_img)

        if SrImage_img.shape[1] == 3:
            SrImage_img_ocr = SrImage_img.mean(dim=1, keepdim=True)
        loss_ocr = ocr_processor.process(SrImage_img_ocr, ocr_label)

        # Calculate the total loss
        loss = (img_loss * initial_img_weight + loss_ocr * initial_ocr_weight) / accumulation_steps
        loss.backward()

        img_loss_list.append(img_loss.item())
        mean_loss_ocr_list.append(loss_ocr.item())

        if iteration % accumulation_steps == 0:
            # torch.nn.utils.clip_grad_norm_(netSR.parameters(), max_norm=3.0)
            optimizerG.step()
            optimizerG.zero_grad()

        # Print progress information
        print(f"Epoch [{epoch}], Iteration [{iteration}/{len(data_loader)}], "
              f"Image Loss: {img_loss.item():.4f}, OCR Loss: {loss_ocr.item():.4f}, "
              f"Total Loss (scaled): {loss.item():.4f}")

    # Handle leftover gradients if total iters not divisible by accumulation_steps
    if iteration % accumulation_steps != 0:
        # torch.nn.utils.clip_grad_norm_(netSR.parameters(), max_norm=3.0)
        optimizerG.step()
        optimizerG.zero_grad()

    metrics['img_loss'].append(np.mean(img_loss_list))
    metrics['ocr_loss'].append(np.mean(mean_loss_ocr_list))
    print_metrics_table(metrics, epoch, parent_folder)

def save_checkpoint(model, epoch, parent_folder):
    checkpoint_dir = os.path.join(parent_folder, "srgan_checkpoint111")
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_out_path = os.path.join(checkpoint_dir, f"srgan_model_epoch_{epoch}.pth")
    state = {"epoch": epoch, "model": model}
    torch.save(state, model_out_path)
    print(f"Checkpoint saved to {model_out_path}")

def print_metrics_table(metrics, epoch, parent_folder):
    save_path = os.path.join(parent_folder, 'metrics_epoch_2111.csv')
    plot_save_path = os.path.join(parent_folder, 'metrics_plot_epoch_2111.png')

    # Create a DataFrame from the metrics dictionary
    metrics_df = pd.DataFrame(metrics)

    # Ensure that all columns are displayed
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.expand_frame_repr', False)  # Prevent wrapping

    # Print the table with epoch information
    print(f"\nMetrics at the end of Epoch {epoch}:\n")
    print(metrics_df)

    metrics_df.to_csv(save_path, index=False)

    # Number of metrics to plot
    num_metrics = len(metrics_df.columns)

    # Set the layout to two columns and remaining rows
    num_cols = 2
    num_rows = (num_metrics + 1) // 2  # Calculate the number of rows needed

    # Create subplots with the specified layout
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 5 * num_rows))
    axs = axs.flatten()  # Flatten the 2D array of axes for easier indexing

    # Plot each metric on a separate subplot
    for i, metric in enumerate(metrics_df.columns):
        axs[i].plot(metrics_df[metric], label=metric, marker='o')
        axs[i].set_title(f'{metric} over Epochs')
        axs[i].set_xlabel('Epoch')
        axs[i].set_ylabel(metric)
        axs[i].legend()
        axs[i].grid(True)

    # Remove unused subplots (if any)
    for i in range(num_metrics, len(axs)):
        fig.delaxes(axs[i])

    # Adjust layout and save the plot
    plt.tight_layout()  # Adjust spacing between subplots
    plt.savefig(plot_save_path)

    # Show the plot
    # plt.show()

def total_gradient(parameters):
    """Computes the gradient norm."""
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    totalnorm = sum((p.grad.data.norm() ** 2) for p in parameters) ** 0.5
    return totalnorm

def test_one_image(netSR, epoch, parent_folder):
    im_input_org = cv2.imread("tets_images/test.png")

    # Determine the device (CUDA if available, else CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move the model to the determined device
    netSR = netSR.to(device)

    # Prepare the input image
    im_input = im_input_org.transpose(2, 0, 1)
    im_input = im_input.reshape(1, *im_input.shape)
    im_input = torch.from_numpy(im_input / 255.).float().to(device)  # Move input to device

    # Forward pass through the generator (SR)
    out = netSR(im_input)

    # Move output to CPU and convert to numpy array
    im_h = out.detach().cpu().numpy()[0].astype(np.float32)
    im_h = im_h * 255.
    im_h = np.clip(im_h, 0, 255)  # Clip values to [0, 255]
    im_h = im_h.transpose(1, 2, 0)

    # Plotting the results
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.resize(im_input_org, (512, 128)))
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.resize(im_h.astype(np.uint8), (512, 128)))
    plt.title("Super-Resolved Image")
    plt.axis('off')

    # Save the results
    output_dir = os.path.join(parent_folder, "output_test_result")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/out_test{epoch}.png")
    plt.savefig("out_test.png")
    plt.close()  # Close the figure to free memory
# Additional functions like train, save_checkpoint, test_one_image, etc. would remain largely unchanged

if __name__ == "__main__":
    main()
