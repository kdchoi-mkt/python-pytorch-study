# The class is used to explain `torch-based model`
import torch
import torch.nn as nn


class BaseLRP(object):
    """The module is implication of Layer-wise Relevance Propagation, which is a kind of explainable AI.
    When the LRP object is used, the model information should be added.
    Furthermore, the object is compatible with `torch`. Therefore if you use tensorflow or keras, you won't use the class directly.
    """

    def __init__(self, simple_model: nn.Module):
        self.model = simple_model

    def calculate(self, input) -> torch.Tensor:
        return self.model(input)

    def explain(self, input) -> torch.Tensor:
        return self._layerwise_relevance_propagation(
            self.model, input, self.calculate(input)
        )

    def _layerwise_relevance_propagation(
        self, model, start_layer_input, end_layer_relevance, epsilon=0.005
    ) -> torch.Tensor:
        """
        Description
        ===========
        Start Layer Input: N dim tensor
        End Layer Relevance: M dim tensor

        Parameter Calculation
        =====================
        Weight: M x N dim Tensor
        Redistribution: N x M dim Tensor
        """

        weight = model.weight
        redistribution = weight * start_layer_input

        redist_weight_avg = self._redist_weight_avg(redistribution, epsilon)

        redistribution /= redist_weight_avg
        redistribution = redistribution.transpose(0, 1)

        start_layer_relevance = redistribution @ end_layer_relevance

        return start_layer_relevance

    def _redist_weight_avg(self, redistribution, epsilon):
        redist_weight_avg = redistribution.sum(dim=1).unsqueeze(1)
        return redist_weight_avg + torch.sign(redist_weight_avg) * epsilon
