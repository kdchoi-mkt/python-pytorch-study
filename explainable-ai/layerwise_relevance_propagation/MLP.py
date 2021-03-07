import torch
import torch.nn as nn
import torch.optim as optim


class MLP(nn.Module):
    """The MLP module is to derive prediction model.
    Therefore the output layer has only two activation: [prob_0, prob_1].
    """

    def __init__(self, input_variable_num):
        self.model = nn.Sequential(
            nn.Linear(input_variable_num, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 2),
            nn.LeakyReLU(),
            nn.Softmax()
        )

    def forward(self, input):
        return self.model(input)