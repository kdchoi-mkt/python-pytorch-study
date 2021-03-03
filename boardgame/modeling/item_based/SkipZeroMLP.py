# Implemented by pyTorch framework
import torch
import torch.nn as nn


class SkipZeroMLP(nn.Module):
    """The model is purposed to avoid missing value problem in recommendation system.

    When user did not check product P_1's preference even if he like it,
    the optimizer and loss function learn the user dislike P_1.
    Therefore, before the end of the calculation, the module filters the value when it is originally 0.
    If you want to treat non-filtered outcome, you should extract `self.model` instance from the `SkipZeroMLP`.
    """

    def __init__(self, input_size, output_size, original_result):
        super().__init__()
        self.original_result = original_result
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 32),
            nn.LeakyReLU(),
            nn.Linear(32, output_size),
        )

    def forward(self, input):
        """The model also has `original` argument to exclude missing data on loss calculation"""
        non_missing_output = (self.original_result > 0) * 1
        return self.model(input) * non_missing_output
