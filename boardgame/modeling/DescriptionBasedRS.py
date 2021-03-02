# Main Recommendation System Inherited Class
from RecommendationSystem import ItemBasedRS
from sklearn.feature_extraction.text import TfidfVectorizer

# Type Hinting
import pandas as pd
import numpy as np

# * Item Description Based Recommendation System
class DescriptionBasedRS(ItemBasedRS, TfidfVectorizer):
    """Description Based Recommendation System Module
    The purpose of the model is to construct recommendation system by boardgame's description.
    Because the description based RS requires the TF-IDF vectorization,
    the model inherits TfidfVectorizer class from sklearn package.

    Because the item-based RS has the main problem, the interests are different for each people,
    another recommendation system like Item 2 Vec, Implicit Feedback, ... will have more power.
    """

    def __init__(
        self,
        data_frame: pd.DataFrame,
        description_column: str,
        name_column: str,
        **kwarg,
    ) -> None:
        self.name_col = name_column
        self.description_col = description_column

        TfidfVectorizer.__init__(self, **kwarg)
        ItemBasedRS.__init__(self, base_data_frame=data_frame)

    def generate_recommend_matrix(self, recommend_base_df):
        return self.fit_transform(recommend_base_df[self.description_col]).toarray()

    def construct_data_frame(self):
        """The input data frame has distinct item.
        Therefore, we do not need to treat duplicate information.
        """
        data_frame = self.base_data_frame[[self.name_col]].reset_index()
        data_frame.columns = ["label_encoder", "name"]

        return data_frame.set_index("label_encoder")

    def _find_index(self, name: str) -> int:
        if name not in self.base_data_frame[self.name_col].to_list():
            raise ValueError(
                f"The boardgame name {name} does not in the database. Try again!"
            )

        target_data = self.base_data_frame.query(f"{self.name_col} == '{name}'")

        return target_data.index[0]
