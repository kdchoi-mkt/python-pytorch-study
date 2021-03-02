# This Item 2 Vec mainly uses pytorch
import torch
import torch.nn as nn
import torch.optim as optim

# Type Hinting
import pandas as pd

# Procedure Visualization module
from tqdm.auto import tqdm

# Data Labeling Module
from sklearn.preprocessing import LabelEncoder


class Item2Vec(object):
    """The recommendation system is also limited to the item-based Recommendation System.
    However, the system generates `meaningful vector` embedded on N dimensional vector space based on the users' played history."""

    def __init__(self, sequence_list, dimension=5, window=3, iteration=1000):
        self.sequence_list = sequence_list
        self.dimension = dimension
        self.window = window
        self.iteration = iteration

        self.item_list = self._to_item_list(sequence_list)
        self.label_encoder = self._label_encode()

        self.item_size = len(self.item_list)

    def continuous_bag_of_items(self):
        """The algorithm is from CBOW(Continuous bag of words).
        In fact, the function must be fixed if we can."""
        batch_size = sum([len(seq) for seq in self.sequence_list])

        item_tensor = torch.zeros(batch_size, self.item_size)
        center_tensor = torch.zeros(batch_size)
        non_zero_item = list()

        total_index = 0
        for sequence in self.sequence_list:
            for index, center_item in enumerate(sequence):
                near_item = self._extract_near_element(sequence, index, self.window)
                current_index = total_index + index

                near_item_index = self.label_encoder.transform(near_item)
                center_item_index = self.label_encoder.transform([center_item])

                for near_index in near_item_index:
                    item_tensor[total_index + index][near_index] += 1

                item_tensor[current_index] /= len(near_item_index)

                if len(near_item_index) != 0:
                    non_zero_item.append(current_index)

                center_tensor[current_index] = int(center_item_index)

            total_index += len(sequence)
        print(f"소실된 자료: {total_index  - len(non_zero_item)}")
        return item_tensor[non_zero_item], center_tensor[non_zero_item].long()

    def training_data(self, model, input_tensor, output_tensor, **optimize_condition):
        validation_ftn = nn.CrossEntropyLoss()
        optimization = optim.Adam(model.parameters(), **optimize_condition)

        progress_bar = tqdm(range(self.iteration))

        for _ in progress_bar:
            predict_element = model(input_tensor)

            cost = validation_ftn(predict_element, output_tensor)

            optimization.zero_grad()
            cost.backward()
            optimization.step()

            progress_bar.set_postfix({"cost": cost})

        return model

    def item_to_vector(self, optimize_condition: dict = dict()):
        """Run Item2Vec Algorithm.

        First, make input tensor and output tensor by CBOW method (`continuous_bag_of_item()`).
        Second, define two layer so that item_size -> embed_dim -> item_size. Note that there are no any activation function.
        Finally, use Softmax to make final report as probability. However, the calculation is automatically conducted by `torch.CrossEntropyLoss`
        """
        self.input_to_output = nn.Sequential(
            nn.Linear(self.item_size, self.dimension),
            nn.Linear(self.dimension, self.item_size),
        )

        input_tensor, output_tensor = self.continuous_bag_of_items()

        self.input_to_output = self.training_data(
            self.input_to_output,
            input_tensor,
            output_tensor,
            **optimize_condition,
        )

        return self.input_to_output

    def _label_encode(self):
        self.label_encoder = LabelEncoder().fit(self.item_list)
        return self.label_encoder

    def _to_item_list(self, sequence_list):
        if type(sequence_list) == pd.Series:
            return list(set(sequence_list.sum()))

        initial_list = []
        for elem in sequence_list:
            initial_list += elem

        return list(set(initial_list))

    def _extract_near_element(self, iterable_obj: list, center, range):
        """Returm iterable_obj[center - range, center + range] without center.
        The iterable_obj MUST support concat operator as `+`"""
        low = center - range
        high = center + range

        if low < 0:
            low = 0
        if high > len(iterable_obj):
            high = len(iterable_obj)

        return iterable_obj[low:center] + iterable_obj[center + 1 : high + 1]
