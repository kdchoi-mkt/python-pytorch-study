# Inherited from LinearLRP
# For the general fully connected AI, the Linear LRP is the basic LRP.
from .LinearLRP import LinearLRP

# Using torch framework.
import torch


class DetailedLinearLRP(LinearLRP):
    def __init__(self, model, output_dim):
        self.model = model
        self.output_dim = output_dim

    def explain(self, input, label, epsilon=0.005):
        one_hot_coding = torch.zeros(self.output_dim)
        one_hot_coding[label] = 1

        layer_value_list, relevance = self._append_forward_propagation_value(input)
        relevance = relevance * one_hot_coding

        return self._calculate_backward_propagation_relevance(
            relevance, layer_value_list, self.model, epsilon
        )
