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
        self.device = device
        self.crossEntropy_loss = nn.CrossEntropyLoss(
            ignore_index=1,  # Assuming 1 is your padding token
            label_smoothing=0.1
        ).to(self.device)
        self.mse_criterion = nn.MSELoss(reduction='mean').cuda()
        # Add a flag to ensure the debug prints only happen once per run
        self._print_flag = False

    def process(self, sr_image, ocr_label):
        # --- 1. ENCODING STEP ---
        text, length = self.converter.encode(ocr_label, batch_max_length=35)

        if self._print_flag:
            print("\n" + "=" * 25 + " OCR LOSS DEBUG (First Batch) " + "=" * 25)
            print(f"[1. ENCODE] Original Labels (Sample 0): '{ocr_label[0]}'")
            print(f"[1. ENCODE] Encoded Text Shape: {text.shape}")
            print(f"[1. ENCODE] Encoded Text Tensor (Sample 0):\n{text[0]}")
            print(f"[1. ENCODE] Sequence Lengths Tensor: {length}")
            print("-" * 70)

        # --- 2. FORWARD PASS ---
        # Input to OCR model is the target sequence, excluding the last token ([EOS])
        # This is standard "teacher forcing"
        input_for_ocr = text[:, :-1]
        preds_sr, feature_sr = self.netOCR(sr_image, input_for_ocr)

        if self._print_flag:
            print(f"[2. FORWARD] Input Shape for OCR Model: {input_for_ocr.shape}")
            print(f"[2. FORWARD] Predictions Shape (Batch, SeqLen, Classes): {preds_sr.shape}")
            # Let's see the prediction for the first character of the first item
            first_char_pred_logits = preds_sr[0, 0, :]
            pred_confidence, pred_index = torch.max(F.softmax(first_char_pred_logits, dim=-1), dim=-1)

            # --- FIXED ---
            # Directly access the character from the converter instead of using the batch-decode function
            # This avoids the IndexError.
            try:
                pred_char = self.converter.character[pred_index.item()]
            except IndexError:
                pred_char = "[UNK]"  # Handle potential out-of-bounds index

            print(
                f"[2. FORWARD] Sample Prediction (1st char): '{pred_char}' with {pred_confidence.item():.2%} confidence")
            print("-" * 70)

        # --- 3. TARGET PREPARATION ---
        # The target for the loss is the original sequence, excluding the first token ([GO])
        target = text[:, 1:]

        if self._print_flag:
            print("[3. TARGET] Creating target by shifting encoded text.")
            print(f"[3. TARGET] Initial Target Shape: {target.shape}")
            print(f"[3. TARGET] Target Tensor (Sample 0) before replacement:\n{target[0]}")

        # Replace the [GO] token (index 0) with the ignore_index token (index 1)
        target_after_replace = torch.where(target == 0, torch.tensor(1, device=target.device), target)

        if self._print_flag:
            print("[3. TARGET] Replaced token 0 with 1 (the ignore_index).")
            print(f"[3. TARGET] Target Tensor (Sample 0) after replacement:\n{target_after_replace[0]}")
            print("-" * 70)

        # --- 4. LOSS CALCULATION ---
        # Reshape for CrossEntropyLoss, which expects (Batch, Classes, SeqLen)
        preds_for_loss = preds_sr.permute(0, 2, 1)

        if self._print_flag:
            print("[4. LOSS] Permuting predictions for loss function.")
            print(f"[4. LOSS] Predictions Shape for Loss Fn: {preds_for_loss.shape}")
            print(f"[4. LOSS] Target Shape for Loss Fn:      {target_after_replace.shape}")

        # Compute the loss
        loss = self.crossEntropy_loss(preds_for_loss, target_after_replace)

        if self._print_flag:
            print(f"\n[4. LOSS] ==> Final Calculated Loss (scalar): {loss.item():.6f}")
            print("=" * 72 + "\n")
            # Set flag to false to prevent printing for subsequent batches
            self._print_flag = False

        return loss

