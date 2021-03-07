# The model is inherited by DeepLearningRS
from .DeepLearningRS import DeepLearningRS

# Use Confidence MSE Loss.
from ..util import ConfidenceMSELossModule, SkipZeroMLP

# The model uses Deep Learning Framework
import torch
import torch.nn as nn
import torch.optim as optim

# Data Handle
import numpy as np
import pandas as pd


class ImplicitFeedbackRS(DeepLearningRS):
    def __init__(
        self,
        data_frame: pd.DataFrame,
        user_col: str,
        item_col: str,
        value_col: str,
        confidence: int = 40,
        iteration: int = 10000,
    ):
        self.confidence = confidence
        DeepLearningRS.__init__(
            self,
            data_frame=data_frame,
            user_col=user_col,
            item_col=item_col,
            value_col=value_col,
            iteration=iteration,
        )

    def generate_recommend_matrix(self):
        input_tensor = self.derive_user_encoding_tensor()
        output_tensor = self.derive_item_rating_tensor()

        confidence_tensor = output_tensor * self.confidence
        output_tensor = (output_tensor > 0) * 1

        self.cost_module = ConfidenceMSELossModule(confidence_tensor).get_module()

        self.learning_model = SkipZeroMLP(self.user_num, self.item_num, output_tensor)

        return (
            self.training_data(input_tensor, output_tensor)
            .model(input_tensor)
            .detach()
            .numpy()
        )
