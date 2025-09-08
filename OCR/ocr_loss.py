import torch
import torch.nn.functional as F
import torch.nn as nn
# import Levenshtein
import torch
import matplotlib.pyplot as plt


class OCRProcessor:
    def __init__(self, netOCR, converter, device):
        self.netOCR = netOCR
        self.converter = converter
        self.entropy_criterion = nn.BCELoss().cuda()
        self.crossEntropy_loss = torch.nn.CrossEntropyLoss(ignore_index=1).cuda()
        self.mse_criterion = nn.MSELoss(reduction='mean').cuda()
        self.device = device



    def transform_tensor_cuda(self,tensor, max_size=106):
        # Ensure tensor is on the GPU (CUDA)
        tensor = tensor.to(self.device)

        N, cols = tensor.shape  # Get the shape of the input tensor

        # Initialize the output tensor with zeros (N, 36, 106) on GPU
        output = torch.zeros(N, cols, max_size, dtype=torch.float32, device=self.device)

        # Use advanced indexing to avoid loops for faster processing
        row_indices = torch.arange(N, device=self.device).view(-1, 1).repeat(1, cols)
        col_indices = torch.arange(cols, device=self.device).view(1, -1).repeat(N, 1)
        value_indices = tensor.long()  # Use the input tensor values as index

        # Set the corresponding index to 1
        output[row_indices, col_indices, value_indices] = 1

        return output



    def process(self, sr_image, ocr_label):
        # Encode the target OCR labels
        text, length = self.converter.encode(ocr_label, batch_max_length=35)
    
        # Forward pass through OCR (keep gradients)
        preds_sr, feature_sr = self.netOCR(sr_image, text[:, :-1])  
    
        # Target for loss calculation (without [GO] symbol)
        target = text[:, 1:]  
        target = torch.where(
            target == 0, torch.tensor(1, device=target.device), target
        )  
    
        # Reshape for CrossEntropyLoss
        preds_sr = preds_sr.permute(0, 2, 1)  
    
        # Compute the differentiable loss
        loss = self.crossEntropy_loss(preds_sr, target)
    
        return loss




