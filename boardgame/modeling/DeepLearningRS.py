# The model is based on torch
import torch
import torch.nn as nn
import torch.optim as optim

# Data Handle
import numpy as np

# Procedure Visualization
from tqdm.auto import tqdm

# Label Encoding
from sklearn.preprocessing import LabelEncoder


class DeepLearningRS(object):
    """The collaborative filtering can be implemented by one-hidden-layer perceptron with non-activation model.
    This means that the MLP deep learning is the generalization of the collaborative filtering.

    To extend the collaborative filtering into deep learning implementation totally, first I wrote the 3 Layer MLP to replace explicit feedback.
    """

    def __init__(self, data_frame, user_col, item_col, value_col, iteration=1000):
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

    def train_model(self):
        self.model = nn.Sequential(
            nn.Linear(self.user_num, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 32),
            nn.LeakyReLU(),
            nn.Linear(32, self.item_num),
        )

        cost_ftn = nn.MSELoss()
        optimization = optim.Adam(self.model.parameters())

        input_tensor = self.derive_user_encoding_tensor()
        output_tensor = self.derive_item_rating_tensor()

        non_zero_matrix = (output_tensor > 0) * 1
        progress_bar = tqdm(range(self.iteration))

        for _ in progress_bar:
            collaborate = non_zero_matrix * self.model(input_tensor)

            cost = cost_ftn(collaborate, output_tensor)

            optimization.zero_grad()
            cost.backward()
            optimization.step()

            progress_bar.set_postfix({"cost": cost})

        return self.model