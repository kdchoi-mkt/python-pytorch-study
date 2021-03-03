# Type Hinting
import numpy as np
import pandas as pd

# Import Normalizer
from sklearn.preprocessing import Normalizer

# Import Base RecommendationSystem Module
from .BaseRS import BaseRS


class CFBasedRS(BaseRS):
    """CFBasedRS is the basic recommendation system class for Collaborative Filtering based Recommendation System.

    The class supports the following functions:
    1. Users' preference for specific item
    2. Users' preference for Top N

    In the CF based RS, _find_index() does not be used but use `_find_user_index()` and `_find_item_index()`.
    """

    def __init__(self, base_data_frame, user_col, item_col, value_col):
        """Because when BaseRS is initiated, the class generates recommend matrix.
        Therefore, the parent __init__ function should be called after the necessary variables are assigned"""
        self.user_col = user_col
        self.item_col = item_col
        self.value_col = value_col

        BaseRS.__init__(self, base_data_frame=base_data_frame)

    def most_prefer_object(self, user, topn=10) -> pd.DataFrame:
        """Return most prefer object for user from Top n."""
        index = self._find_user_index(user)

        return self._most_prefer_object_by_index(index, topn)

    def get_user_item_prefer(self, user, item) -> str:
        """Return the predicted preference for user"""
        user_index = self._find_user_index(user)
        item_index = self._find_item_index(item)

        return self._get_user_item_prefer_by_index(user_index, item_index)

    def _most_prefer_object_by_index(self, index, topn) -> pd.DataFrame:
        user_vector = self.recommend_matrix[index]

        rank_vector = user_vector.argsort()[-topn:]
        user_vector = user_vector[rank_vector]

        user_data = self.construct_data_frame().iloc[[index], rank_vector].transpose()

        return user_data.sort_values(
            list(user_data.columns), ascending=False
        ).reset_index()

    def _get_user_item_prefer_by_index(self, user_index, item_index) -> float:
        return self.recommend_matrix[user_index][item_index]

    def _find_user_index(self, name: str) -> int:
        """Because each recommendation system class has different motivation and different implementation,
        the base class cannot previously define `index` caller (i.e. the class cannot call index from dataframe.index).
        The `_find_user_index()` function matches the user name and their own user number (0 ~ ...).
        In general, the index has the form of calling `LabelEncoder` from `sklearn.preprocessing`.
        """
        return 0

    def _find_item_index(self, name: str) -> int:
        """Because each recommendation system class has different motivation and different implementation,
        the base class cannot previously define `index` caller (i.e. the class cannot call index from dataframe.index).
        The `_find_item_index()` function matches the item name and their own item number (0 ~ ...).
        In general, the index has the form of calling `LabelEncoder` from `sklearn.preprocessing`.
        """
        return 0