import argparse
import string
import numpy as np
import torch
from torch.utils.data import DataLoader
from collections import OrderedDict
from tqdm import tqdm

# --- Import necessary components from your project ---
# Make sure these paths are correct relative to where you run the script
from datasets.data_set_from_pt import DatasetFromPT
from OCR.model import ModelOCR
from OCR.utils import AttnLabelConverter
from OCR.ocr_loss import OCRProcessor


def main():
    parser = argparse.ArgumentParser(
        description="Calculate OCR loss on High-Resolution Ground Truth Images"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/dataset.pt",
        help="Path to the preprocessed .pt dataset file.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for processing the dataset.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of dataloader workers."
    )
    parser.add_argument(
        "--ocr_model_path",
        type=str,
        default="back_up_models/OCR/TPS-ResNet-BiLSTM-Attn-case-sensitive.pth",
        # default="back_up_models/OCR/TPS-ResNet-BiLSTM-Attn.pth",
        help="Path to the pre-trained OCR model weights.",
    )
    opt = parser.parse_args()
    print("--- OCR Loss Baseline Test on HR Images ---")
    print(opt)

    # --- Setup Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Dataset Loading ---
    print("\n===> Loading dataset...")
    try:
        full_dataset = DatasetFromPT(opt.dataset)
        data_loader = DataLoader(
            dataset=full_dataset,
            num_workers=opt.workers,
            batch_size=opt.batch_size,
            shuffle=False,  # No need to shuffle for evaluation
        )
        print(f"Successfully loaded {len(full_dataset)} images.")
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {opt.dataset}")
        return

    # --- OCR Model Setup (Identical to your training script) ---
    print("\n===> Building and loading OCR model...")
    from argparse import Namespace
    ocr_config = Namespace(
        Transformation='TPS', FeatureExtraction='ResNet', SequenceModeling='BiLSTM',
        Prediction='Attn', num_fiducial=20, input_channel=1, output_channel=512,
        hidden_size=256, num_class=96, imgH=32, imgW=100, batch_max_length=35, device=device
    )
    netOCR = ModelOCR(ocr_config).to(device)

    try:
        state_dict = torch.load(opt.ocr_model_path, map_location=device, weights_only=True)
        # Handle state dicts saved with 'module.' prefix from DataParallel
        new_state_dict = OrderedDict((k[7:] if k.startswith('module.') else k, v) for k, v in state_dict.items())
        netOCR.load_state_dict(new_state_dict)
    except FileNotFoundError:
        print(f"Error: OCR model weights not found at {opt.ocr_model_path}")
        return

    # Set model to evaluation mode
    netOCR.eval()

    # Setup converter and loss processor
    character = string.printable[:-6]
    # character = "0123456789abcdefghijklmnopqrstuvwxyz"

    converter = AttnLabelConverter(character)
    ocr_processor = OCRProcessor(netOCR, converter, device)

    # --- Loss Calculation Loop ---
    print("\n===> Calculating OCR loss on HR images...")
    all_ocr_losses = []

    # Disable gradient calculations for efficiency
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Processing Batches"):
            # We only need the high-res image and the label
            _, real_img, ocr_label = batch

            # Move data to the selected device
            real_img = real_img.to(device)

            # Convert HR image to grayscale, as expected by the OCR model
            # The OCR model expects a single channel (grayscale) input
            hr_image_gray = real_img.mean(dim=1, keepdim=True)

            # Calculate the OCR loss on the ground truth image
            ocr_loss = ocr_processor.process(hr_image_gray, ocr_label)

            all_ocr_losses.append(ocr_loss.item())

    # --- Report Final Results ---
    average_loss = np.mean(all_ocr_losses)

    print("\n" + "=" * 40)
    print("          TEST COMPLETE          ")
    print("=" * 40)
    print(f"Total images processed: {len(full_dataset)}")
    print(f"Average OCR Loss on HR Images: {average_loss:.4f}")
    print("\nThis value is your effective 'best-case' OCR loss.")
    print("Your trained SR model's validation OCR loss should aim to approach this number.")


if __name__ == "__main__":
    main()