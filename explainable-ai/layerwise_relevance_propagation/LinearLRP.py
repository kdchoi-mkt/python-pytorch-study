# The LRP model needs help from torch
import torch
import torch.nn as nn

# import base LRP model
from .BaseLRP import BaseLRP


class LinearLRP(BaseLRP):
    """The module is implication of Layer-wise Relevance Propagation, which is a kind of explainable AI.
    When the LRP object is used, the model information should be added.
    Furthermore, the object is compatible with `torch`. Therefore if you use tensorflow or keras, you won't use the class directly.
    Especially, the model assumed that it is constructed by nn.Sequential which is consisted of activation and Linear, and the model is trained.
    """

    def __init__(self, sequential_model: nn.Module):
        self.model = sequential_model

    def explain(self, input, epsilon=0.005):
        """First, construct input_lists (layer value lists) that includes layers' data for layerwise forward propagation.
        Second, analyze the relevance with LRP method iteratively.
        """
        layer_value_list, relevance = self._append_forward_propagation_value(input)

        return self._calculate_backward_propagation_relevance(
            relevance, layer_value_list, self.model, epsilon
        )

    def _append_forward_propagation_value(self, input) -> tuple:
        layer_value_lists = [input]

        for submodule in self.model:
            if type(submodule) == nn.Linear:
                layer_value_lists.append(input)
            input = submodule(input)

        return layer_value_lists, input

    def _calculate_backward_propagation_relevance(
        self, relevance, layer_value_list, model, epsilon
    ):
        for submodule in reversed(model):
            if type(submodule) == nn.Linear:
                layer_value = layer_value_list.pop()
                relevance = self._layerwise_relevance_propagation(
                    submodule, layer_value, relevance, epsilon=epsilon
                )

        return relevance