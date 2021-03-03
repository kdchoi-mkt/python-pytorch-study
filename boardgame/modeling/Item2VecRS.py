# Recommendation System Module
from RecommendationSystem import ItemBasedRS
from util import Item2Vec

# Data Treatment
import pandas as pd
import numpy as np


class Item2VecRS(ItemBasedRS, Item2Vec):
    def __init__(self, data_frame, sequence_col, dimension=5, window=3, iteration=1000):
        Item2Vec.__init__(self, data_frame[sequence_col], dimension, window, iteration)
        ItemBasedRS.__init__(self, data_frame)

    def generate_recommend_matrix(self) -> np.array:
        model = self.item_to_vector()
        return model[0].weight.T.detach().numpy()

    def construct_data_frame(self) -> pd.DataFrame:
        item_df = pd.DataFrame(
            data=[self.label_encoder.classes_], index=["name"]
        ).transpose()
        item_df["label_encoder"] = self.label_encoder.transform(item_df["name"])
        item_df = item_df.set_index(["label_encoder"])

        item_vector_df = pd.DataFrame(self.recommend_matrix)
        item_vector_df.columns = "Dimension_" + item_vector_df.columns.astype(str)

        return item_df.join(item_vector_df)

    def _find_index(self, name: str) -> int:
        return self.label_encoder.transform([name])[0]


if __name__ == "__main__":
    item_2_vec = Item2Vec([[1, 1, 1, 1], [1, 2, 3], [4, 5]])
    print(item_2_vec.item_to_vector())