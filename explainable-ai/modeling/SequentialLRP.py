# The LRP model needs help from torch
import torch
import torch.nn as nn

# import base LRP model
from .BaseLRP import BaseLRP


class SequentialLRP(BaseLRP):
    """The module is implication of Layer-wise Relevance Propagation, which is a kind of explainable AI.
    When the LRP object is used, the model information should be added.
    Furthermore, the object is compatible with `torch`. Therefore if you use tensorflow or keras, you won't use the class directly.
    Especially, the model assumed that it is constructed by nn.Sequential, and the model is trained.
    """

    def __init__(self, sequential_model: nn.Module):
        self.model = sequential_model

    def explain(self, input, epsilon=0.005):
        """First, construct input_lists that includes input data for layerwise forward propagation.
        Second, analyze the relevance with LRP method iteratively.
        """
        input_lists = [input]

        for submodule in self.model:
            if type(submodule) == nn.Linear:
                input_lists.append(input)
            input = submodule(input)

        relevance = input

        for submodule in reversed(self.model):
            if type(submodule) == nn.Linear:
                start_layer_input = input_lists.pop()
                relevance = self._layerwise_relevance_propagation(
                    submodule, start_layer_input, relevance, epsilon=epsilon
                )

        return relevance
