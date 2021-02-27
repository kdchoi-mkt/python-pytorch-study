# Main Recommendation System Inherited Class
from sklearn.feature_extraction.text import TfidfVectorizer

# Type Hinting
import pandas as pd
import numpy as np

# * Item Description Based Recommendation System
class DescriptionBasedRS(TfidfVectorizer):
    """Description Based Recommendation System Module
    The purpose of the model is to construct recommendation system by boardgame's description.
    Because the description based RS requires the TF-IDF vectorization,
    the model inherits TfidfVectorizer class from sklearn package.

    Because the item-based RS has the main problem, the interests are different for each people,
    another recommendation system like Item 2 Vec, Implicit Feedback, ... will have more powerful
    """

    def __init__(
        self,
        data_frame: pd.DataFrame,
        description_column: str,
        name_column: str,
        **kwarg,
    ) -> None:
        super().__init__(**kwarg)
        self.data_frame = data_frame
        self.name_col = name_column
        self.description_col = description_column
        self.vectorized_form = self.fit_transform(data_frame[description_column])

    def similarity_two_object(self, name_1: str, name_2: str) -> np.array:
        """Derive COSINE similarity between two objects."""
        index_1 = self._find_index(name_1)
        index_2 = self._find_index(name_2)

        return self._similarity_two_object_by_index(index_1, index_2)

    def most_similar_object(
        self, name: str, topn: int = 10, exclude_self: bool = False
    ) -> pd.DataFrame:
        index = self._find_index(name)

        return self._most_similar_object_by_index(index, topn, exclude_self)

    def _similarity_two_object_by_index(self, index_1: int, index_2: int) -> np.array:
        vector_1 = self.vectorized_form[index_1].toarray()
        vector_2 = self.vectorized_form[index_2].toarray()

        return vector_1 @ vector_2.T

    def _most_similar_object_by_index(
        self, index: int, topn: int = 10, exclude_self: bool = False
    ) -> pd.DataFrame:
        data_frame = self.data_frame.reset_index(drop=True)
        vector = self.vectorized_form[index].toarray()

        similar_vector = self.vectorized_form @ vector.T
        similar_vector = similar_vector.reshape(-1)

        if exclude_self:
            similar_vector[index] = -1

        rank_vector = similar_vector.argsort()[-topn:]
        similar_vector = similar_vector[rank_vector]

        similar_df = data_frame.iloc[rank_vector, :]
        similar_df["similarity"] = similar_vector

        return similar_df.sort_values(["similarity"], ascending=False)

    def _find_index(self, name: str) -> int:
        if name not in self.data_frame[self.name_col].to_list():
            raise ValueError(
                f"The boardgame name {name} does not in the database. Try again!"
            )

        target_data = self.data_frame.query(f"{self.name_col} == '{name}'")
        return target_data.index[0]
