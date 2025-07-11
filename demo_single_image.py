import argparse
import cv2
import torch
from torch.autograd import Variable
import numpy as np
import time

# Load both models
model1 = torch.load(R"D:\Researches\SR\FirstPaperCode\backup_models\RES\2025-04-09_13-13-06\srgan_checkpoint111\srgan_model_epoch_491.pth", map_location=torch.device('cpu'))["model"]  # SRGAN
  # Your model

# Read input image and ground truth
im_input_org = cv2.imread("lrTZtesteasy_736.png")
GT = im_input_org.copy()

# Preprocess the input image
im_input = im_input_org.transpose(2, 0, 1)
im_input = im_input.reshape(1, im_input.shape[0], im_input.shape[1], im_input.shape[2])
im_input = torch.from_numpy(im_input / 255.).float()

# Run inference with SRGAN (model1)
model1 = model1.cpu()
start_time = time.time()
out_srgan = model1(im_input)
elapsed_time_srgan = time.time() - start_time
out_srgan = out_srgan.cpu()

# Postprocess SRGAN output
im_h_srgan = out_srgan.data[0].numpy().astype(np.float32)
im_h_srgan = im_h_srgan * 255.
im_h_srgan[im_h_srgan < 0] = 0
im_h_srgan[im_h_srgan > 255.] = 255.
im_h_srgan = im_h_srgan.transpose(1, 2, 0)





# Display and save outputs
cv2.imshow("Output SRGAN", cv2.resize(im_h_srgan.astype(np.uint8), (512, 128)))
cv2.imwrite("output_srgan.png", cv2.resize(im_h_srgan.astype(np.uint8), (512, 128)))


# Display and save the input and ground truth images
cv2.imshow("Input", cv2.resize(im_input_org, (512, 128)))
cv2.imwrite("input.png", cv2.resize(im_input_org, (512, 128)))



cv2.waitKey(0)
