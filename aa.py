import torch
import torch.nn.functional as F
import torch.nn as nn
import Levenshtein

class OCRProcessor:
    def __init__(self, netOCR, converter, use_cuda):
        self.netOCR = netOCR
        self.converter = converter
        self.entropy_criterion = nn.BCELoss().cuda()
        self.mse_criterion = nn.MSELoss(reduction='mean').cuda()
        self.device = torch.device("cuda" if use_cuda else "cpu")

    def compute_loss(self, generated_text, real_text):
        """
        Compute the OCR loss using a placeholder for non-differentiable loss.
        """
        # If you cannot differentiate Levenshtein distance, use another metric if possible
        # For now, this function does not participate in the gradient computation
        return Levenshtein.distance(generated_text, real_text) / len(generated_text)

    def process(self, fake, real, ocr_label, batchSize):
        length_for_pred = torch.IntTensor([20] * batchSize).to(self.device)
        text_for_pred = torch.LongTensor(batchSize, 20 + 1).fill_(0).to(self.device)

        preds, feature = self.netOCR(fake, text_for_pred, is_train=False)
        preds_real, feature_real = self.netOCR(real, text_for_pred, is_train=False)

        loss_features = self.mse_criterion(feature, feature_real)

        _, preds_index = preds.max(2)

        preds_str = self.converter.decode(preds_index, length_for_pred)
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)

        total_loss_ocr = torch.tensor(0.0, device=self.device)
        total_loss_confidence = torch.tensor(0.0, device=self.device)
        num_samples = 0

        for pred_ocr, pred_max_prob, real_ocr in zip(preds_str, preds_max_prob, ocr_label):
            pred_ocr = pred_ocr.replace('[s]', '')
            confidence_score = pred_max_prob.cumprod(dim=0)[-1]
            loss_ocr = torch.tensor(self.compute_loss(pred_ocr, real_ocr), device=self.device, dtype=torch.float32)
            loss_confidence = self.entropy_criterion(confidence_score.unsqueeze(0).to(self.device), torch.tensor([1.0], device=self.device))

            total_loss_ocr += loss_ocr
            total_loss_confidence += loss_confidence
            num_samples += 1

        if num_samples > 0:
            mean_loss_ocr = total_loss_ocr / num_samples
            mean_loss_confidence = total_loss_confidence / num_samples
        else:
            mean_loss_ocr = torch.tensor(0.0, device=self.device, dtype=torch.float32)
            mean_loss_confidence = torch.tensor(0.0, device=self.device, dtype=torch.float32)

        return mean_loss_ocr, mean_loss_confidence, loss_features
