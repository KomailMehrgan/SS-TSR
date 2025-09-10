import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from collections import OrderedDict
import numpy as np

# --- Your imports ---
from model.tsrn import TSRN
from OCR.model import ModelOCR
from OCR.utils import AttnLabelConverter
from OCR.ocr_loss import OCRProcessor
from datasets.data_set_from_pt import DatasetFromPT

# ==== Setup ====
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# --- Dataset: just 4 samples ---
dataset = DatasetFromPT("datasets/dataset.pt")
subset = Subset(dataset, range(4))
loader = DataLoader(subset, batch_size=4, shuffle=False)

# --- SR Model (TSRN as example) ---
netSR = TSRN(scale_factor=2, width=128, height=32, STN=True,
             srb_nums=5, mask=True, hidden_units=64).to(device)

# --- OCR Model ---
from argparse import Namespace
ocr_config = Namespace(Transformation='TPS', FeatureExtraction='ResNet', SequenceModeling='BiLSTM',
                       Prediction='Attn', num_fiducial=20, input_channel=1, output_channel=512,
                       hidden_size=256, num_class=96, imgH=32, imgW=100,
                       batch_max_length=35, device=device)
netOCR = ModelOCR(ocr_config).to(device)

# Load pretrained OCR weights
state_dict = torch.load("back_up_models/OCR/TPS-ResNet-BiLSTM-Attn-case-sensitive.pth",
                        map_location=device, weights_only=True)
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] if k.startswith('module.') else k
    new_state_dict[name] = v
netOCR.load_state_dict(new_state_dict)

# Freeze OCR parameters
for p in netOCR.parameters():
    p.requires_grad = False
netOCR.train()

# Converter + OCR Processor
import string
character = string.printable[:-6]
converter = AttnLabelConverter(character)
ocr_processor = OCRProcessor(netOCR, converter, device)

# Loss
mse_criterion = nn.MSELoss().to(device)

# --- Get one batch ---
batch = next(iter(loader))
input_img, real_img, ocr_label = batch
input_img, real_img = input_img.to(device), real_img.to(device)

# Helper: run forward/backward once
def run_case(use_ocr: bool, weight: float):
    netSR.zero_grad()
    sr_out = netSR(input_img)
    sr_rgb = sr_out[:, :3, :, :] if sr_out.shape[1] == 4 else sr_out
    img_loss = mse_criterion(sr_rgb, real_img)

    if use_ocr:
        sr_gray = sr_rgb.mean(dim=1, keepdim=True)
        ocr_loss = ocr_processor.process(sr_gray, ocr_label)
        total_loss = img_loss + weight * ocr_loss
    else:
        total_loss = img_loss

    total_loss.backward()
    grads = [p.grad.clone() for p in netSR.parameters() if p.grad is not None]
    return grads

# === CASE 1: OCR weight = 0 ===
grad_case1 = run_case(use_ocr=True, weight=0.0)

# === CASE 2: No OCR at all ===
grad_case2 = run_case(use_ocr=False, weight=0.0)

# === CASE 3: OCR weight > 0 (e.g. 0.01) ===
grad_case3 = run_case(use_ocr=True, weight=0.0001)

# --- Compare ---
diffs12 = [torch.norm(g1 - g2).item() for g1, g2 in zip(grad_case1, grad_case2)]
diffs13 = [torch.norm(g1 - g3).item() for g1, g3 in zip(grad_case1, grad_case3)]

# print("Per-parameter gradient diff (Case1 vs Case2):", diffs12)
print("Total difference norm (Case1 vs Case2):", np.sum(diffs12))

# print("Per-parameter gradient diff (Case1 vs Case3):", diffs13)
print("Total difference norm (Case1 vs Case3):", np.sum(diffs13))

# Also print gradient norms to see OCR effect
norms1 = [torch.norm(g).item() for g in grad_case1]
norms3 = [torch.norm(g).item() for g in grad_case3]
print("Gradient norms (Case1, ocr_weight=0):", norms1)
print("Gradient norms (Case3, ocr_weight=0.01):", norms3)
