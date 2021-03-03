# The model is based on torch
import torch
import torch.nn as nn
import torch.optim as optim

# Data Handle
import numpy as np
import pandas as pd

# Procedure Visualization
from util import TrainVisualize

# Label Encoding
from sklearn.preprocessing import LabelEncoder


class SkipZeroMLP(nn.Module):
    """The model is purposed to avoid missing value problem in recommendation system.

    When user did not check product P_1's preference even if he like it,
    the optimizer and loss function learn the user dislike P_1.
    Therefore, before the end of the calculation, the module filters the value when it is originally 0.
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


class DeepLearningRS(TrainVisualize):
    """The collaborative filtering can be implemented by one-hidden-layer perceptron with non-activation model.
    This means that the MLP deep learning is the generalization of the collaborative filtering.

    To extend the collaborative filtering into deep learning implementation totally,
    first I wrote the 3 Layer MLP to replace explicit feedback.
    """

    def __init__(
        self,
        data_frame: pd.DataFrame,
        user_col: str,
        item_col: str,
        value_col: str,
        iteration: int = 10000,
    ):
        self.data_frame = data_frame
        self.user_col = user_col
        self.item_col = item_col
        self.value_col = value_col
        self.iteration = iteration

        self.label_encoding = LabelEncoder().fit(self.data_frame[user_col])

        self.user_num = self.data_frame[self.user_col].nunique()
        self.item_num = self.data_frame[self.item_col].nunique()

    def derive_user_encoding_tensor(self):
        """This is used to input tensor"""

        return torch.ones(self.user_num).diag()

    def derive_item_rating_tensor(self):
        """This is used to output tensor"""

        rating_data = self.data_frame.pivot(
            columns=self.item_col, index=self.user_col, values=self.value_col
        ).fillna(0)

        return torch.Tensor(np.array(rating_data))

    def generate_recommend_matrix(self):
        input_tensor = self.derive_user_encoding_tensor()
        output_tensor = self.derive_item_rating_tensor()

        self.learning_model = SkipZeroMLP(self.user_num, self.item_num, output_tensor)
        self.optimize_module = optim.Adam
        self.cost_module = nn.MSELoss

        return self.training_data(input_tensor, output_tensor).model(input_tensor)