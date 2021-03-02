# The TrainVisualize class is for torch framework
import torch

# Procedure Visualization
from tqdm.auto import tqdm


class TrainVisualize(object):
    def __init__(self, iteration, learning_model, optimize_module, cost_module):
        self.iteration = iteration
        self.learning_model = learning_model
        self.optimize_module = optimize_module
        self.cost_module = cost_module

    def training_data(
        self,
        input_tensor: torch.Tensor,
        output_tensor: torch.Tensor,
        **optimize_condition
    ):
        """If you want to use `training_data()`, you should assign
        1. self.optimize_module
        2. self.cost_module
        3. self.learning_model
        """
        optimize = self.optimize_module(
            self.learning_model.parameters(), **optimize_condition
        )
        cost_ftn = self.cost_module()

        progress_bar = tqdm(range(self.iteration))

        for _ in progress_bar:
            predict_element = self.learning_model(input_tensor)

            cost = cost_ftn(predict_element, output_tensor)

            optimize.zero_grad()
            cost.backward()
            optimize.step()

            progress_bar.set_postfix({"cost": cost})

        return self.learning_model