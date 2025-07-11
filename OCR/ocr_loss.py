import torch
import torch.nn.functional as F
import torch.nn as nn
# import Levenshtein
import torch
import matplotlib.pyplot as plt


class OCRProcessor:
    def __init__(self, netOCR, converter, use_cuda):
        self.netOCR = netOCR
        self.converter = converter
        self.entropy_criterion = nn.BCELoss().cuda()
        self.crossEntropy_loss = torch.nn.CrossEntropyLoss(ignore_index=1).cuda()
        self.mse_criterion = nn.MSELoss(reduction='mean').cuda()
        self.device = torch.device("cuda" if use_cuda else "cpu")

    def compute_loss(self, generated_text, real_text):
        """
        Compute the OCR loss using a placeholder for non-differentiable loss.
        """
        # If you cannot differentiate Levenshtein distance, use another metric if possible
        # For now, this function does not participate in the gradient computation
        return Levenshtein.distance(generated_text, real_text) / len(generated_text)

    def transform_tensor_cuda(self,tensor, max_size=106):
        # Ensure tensor is on the GPU (CUDA)
        tensor = tensor.to('cuda')

        N, cols = tensor.shape  # Get the shape of the input tensor

        # Initialize the output tensor with zeros (N, 36, 106) on GPU
        output = torch.zeros(N, cols, max_size, dtype=torch.float32, device='cuda')

        # Use advanced indexing to avoid loops for faster processing
        row_indices = torch.arange(N, device='cuda').view(-1, 1).repeat(1, cols)
        col_indices = torch.arange(cols, device='cuda').view(1, -1).repeat(N, 1)
        value_indices = tensor.long()  # Use the input tensor values as index

        # Set the corresponding index to 1
        output[row_indices, col_indices, value_indices] = 1

        return output



    def process(self, sr_image, ocr_label):
        # Encode the target OCR labels
        text, length = self.converter.encode(ocr_label, batch_max_length=35)

        # Compute the Super-Resolution OCR prediction
        with torch.no_grad():
            preds_sr, feature_sr = self.netOCR(sr_image, text[:, :-1])  # Align with Attention.forward

        # Target for loss calculation (without [GO] symbol)
        target = text[:, 1:]  # Shape: [batch_size, sequence_length]
        target = torch.where(target == 0, torch.tensor(1, device=target.device), target)  # Change 0 to 1 for loss calculation

        # Reshape preds_sr and target for CrossEntropyLoss
        preds_sr = preds_sr.permute(0, 2, 1)  # CrossEntropyLoss expects [batch_size, num_classes, seq_length]



        # Compute the loss
        loss = self.crossEntropy_loss(preds_sr, target)

        return loss




