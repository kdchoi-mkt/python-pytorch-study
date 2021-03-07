import torch
import torch.nn as nn


class ConfidenceMSELossModule(object):
    def __init__(self, confidence_tensor):
        super().__init__()
        self.confidence_tensor = confidence_tensor

    def get_module(self):
        confidence_tensor = self.confidence_tensor

        class ConfidenceMSELoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.confidence_tensor = confidence_tensor

            def forward(self, prediction, real_value):
                mse_loss = (prediction - real_value) ** 2
                return (mse_loss * self.confidence_tensor).sum()

        return ConfidenceMSELoss