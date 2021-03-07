# The model is inherited by CF based RS
from ..base import CFBasedRS

# Procedure Visualization
from ..util import TrainVisualize, SkipZeroMLP

# Assist from torch
import torch
import torch.nn as nn
import torch.optim as optim

# Data Handle
import numpy as np
import pandas as pd

# Label Encoding
from sklearn.preprocessing import LabelEncoder


class DeepLearningRS(CFBasedRS, TrainVisualize):
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
        self.iteration = iteration

        self.user_label_encoding = LabelEncoder().fit(data_frame[user_col])
        self.item_label_encoding = LabelEncoder().fit(data_frame[item_col])

        self.user_num = data_frame[user_col].nunique()
        self.item_num = data_frame[item_col].nunique()

        self.optimize_module = optim.Adam
        self.cost_module = nn.MSELoss

        CFBasedRS.__init__(
            self,
            base_data_frame=data_frame,
            user_col=user_col,
            item_col=item_col,
            value_col=value_col,
        )

    def generate_recommend_matrix(self):
        input_tensor = self.derive_user_encoding_tensor()
        output_tensor = self.derive_item_rating_tensor()
        
        self.learning_model = SkipZeroMLP(self.user_num, self.item_num, output_tensor)

        return (
            self.training_data(input_tensor, output_tensor)
            .model(input_tensor)
            .detach()
            .numpy()
        )

    def construct_data_frame(self):
        data_frame = pd.DataFrame(self.recommend_matrix)
        data_frame.index = self.user_label_encoding.inverse_transform(data_frame.index)
        data_frame.index = pd.Series(data_frame.index, name="user")

        data_frame.columns = self.item_label_encoding.inverse_transform(
            data_frame.columns
        )
        data_frame.columns = pd.Series(data_frame.columns, name="item")

        return data_frame

    def derive_user_encoding_tensor(self):
        """This is used to input tensor"""
        return torch.ones(self.user_num).diag()

    def derive_item_rating_tensor(self):
        """This is used to output tensor.

        Because data_frame.pivot() sorts index and columns automatically, we does not need to treat orders and encoding.
        (In fact, the labeling also works based on the names' order)
        """
        rating_data = self.base_data_frame.pivot(
            columns=self.item_col, index=self.user_col, values=self.value_col
        ).fillna(0)

        return torch.Tensor(np.array(rating_data))

    def _find_user_index(self, name):
        return self.user_label_encoding.transform([name])[0]

    def _find_item_index(self, name):
        return self.item_label_encoding.transform([name])[0]
