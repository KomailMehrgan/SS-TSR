import os
import torch
from torch.utils.data import DataLoader
# We use your existing script to process the images
from dataset_prepration_mask import DatasetFromImages
from tqdm import tqdm
import argparse


def preprocess_and_save(data_path, output_file, batch_size=64):
    """
    Loads a dataset from image folders, processes it, and saves it to a single .pt file.

    Args:
        data_path (str): The root directory of the dataset (e.g., 'datasets/data').
        output_file (str): The path where the processed .pt file will be saved.
        batch_size (int): The batch size for efficient data loading.
    """
    print("===> Loading dataset from image folders...")
    # Initialize the dataset using the folder-based loader (processes 100% of it)
    image_dataset = DatasetFromImages(sr_data_path=data_path, scale=1.0)

    # Use DataLoader for efficient, multi-threaded loading from disk
    data_loader = DataLoader(
        dataset=image_dataset,
        num_workers=1,
        batch_size=batch_size,
        shuffle=True
    )

    print(f"Found {len(image_dataset)} images to process.")

    # Lists to hold all the processed data
    all_lr_tensors = []
    all_hr_tensors = []
    all_labels = []

    print("===> Processing images and labels into tensors...")
    # Use tqdm for a progress bar
    for lr_tensor_batch, hr_tensor_batch, label_batch in tqdm(data_loader, desc="Processing Batches"):
        # Detach tensors from the computation graph and move to CPU to save memory
        all_lr_tensors.extend([t.cpu() for t in lr_tensor_batch])
        all_hr_tensors.extend([t.cpu() for t in hr_tensor_batch])
        all_labels.extend(label_batch)

    print(f"\n===> Saving {len(all_labels)} processed items to {output_file}...")

    # Save the data in a dictionary
    torch.save({
        'lr_images': all_lr_tensors,
        'hr_images': all_hr_tensors,
        'labels': all_labels
    }, output_file)

    print("===> Preprocessing complete! Dataset is ready for training.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dataset Preprocessing to .pt file")
    parser.add_argument("--data_path", type=str, default="./data", help="Path to the root data folder.")
    parser.add_argument("--output_file", type=str, default="./dataset.pt",
                        help="Path to save the output .pt file.")

    args = parser.parse_args()

    # Create the directory for the output file if it doesn't exist
    # This handles cases where the output file is in a subdirectory, e.g., "processed/dataset.pt"
    if os.path.dirname(args.output_file):
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    preprocess_and_save(args.data_path, args.output_file)