import os
import time
import cv2
import torch
import numpy as np

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model_path = "backup_models/RES/2025-04-09_13-13-06/srgan_checkpoint111/srgan_model_epoch_491.pth"
model = torch.load(model_path, map_location=device)["model"].to(device)

# Define input and output folders
input_folder = "OCR/Benchmark_ocr/dataset/TextZoom_testMedium/LR_images"
output_folder = "output_mynet_medium_491"
os.makedirs(output_folder, exist_ok=True)

# Process each image in the input folder
for image_name in os.listdir(input_folder):
    image_path = os.path.join(input_folder, image_name)

    # Read input image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image {image_path}. Skipping.")
        continue

    # Preprocess the image
    image = image.transpose(2, 0, 1)  # Change HWC to CHW
    image = image[np.newaxis, :, :, :]  # Add batch dimension
    image = torch.from_numpy(image / 255.).float().to(device)

    # Run inference
    start_time = time.time()
    output = model(image).cpu()
    elapsed_time = time.time() - start_time

    # Post-process output
    output_image = output.data[0].numpy().astype(np.float32) * 255.
    output_image = np.clip(output_image, 0, 255).transpose(1, 2, 0)  # Change CHW to HWC

    # Save output image
    output_path = os.path.join(output_folder, image_name)
    cv2.imwrite(output_path, output_image.astype(np.uint8))

    print(f"Processed {image_name} in {elapsed_time:.4f} sec: output saved at {output_path}.")
