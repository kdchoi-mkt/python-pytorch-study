# Type Hinting
import numpy as np
import pandas as pd

# Import Normalizer
from sklearn.preprocessing import Normalizer


class ItemBasedRS(object):
    """ItemBased RS is the basic Recommendation System class to implement high-level RS

    The object has the following useful functions:
    1. COSINE Simularity Between Two Items
    2. Top N based on COSINE Simularity
    """

    def __init__(self, base_data_frame: pd.DataFrame):
        self.base_data_frame = base_data_frame
        self.normalizer = Normalizer()
        self.recommend_matrix = self.generate_recommend_matrix(base_data_frame)

    def generate_recommend_matrix(self, recommend_base_df) -> np.array:
        """The function is for generate recommend matrix.
        The matrix can be as TF-IDF matrix, or N-dim embedding matrix.

        Anyway, each row represents the embedded vector of items.
        Note that the vector are not be normalized (i.e. $\\|x_i\\| \\not= 1$ in general.)

        If you inherit the object, you should modify the function.
        Of course, the return value is 2-dim array."""
        return np.zeros(1, 1)

    def most_similar_object(
        self, name: str, topn: int = 10, exclude_self: bool = False
    ) -> pd.DataFrame:
        index = self._find_index(name)

        return self._most_similar_object_by_index(index, topn, exclude_self)

    def get_similarity_bw_object(self, name_1: str, name_2: str) -> np.array:
        """The similarirty is based on Cosine Similarity
        $Sim(a, b) = a\\cdot b / (\\|a\\|\\times\\|b\\|)$
        """
        index_1 = self._find_index(name_1)
        index_2 = self._find_index(name_2)

        return self._get_similarity_bw_object_index(index_1, index_2)

    def construct_data_frame(self):
        """Construct data frame about the recommend matrix.
        The data frame is match table between label encoding result and item.

        If you want, you can add another columns such as description, etc.
        The table has the following format:

        | name | label_encoder | ... |
        |------|---------------|-----|
        |  A   |       0       |     |
        | ...  |      ...      |     |

        Then, set index to `label_encoder`
        """
        data = pd.DataFrame(columns=["name", "label_encoder"])
        return data.set_index("label_encoder")

    def _get_similarity_bw_object_index(self, index_1: int, index_2: int) -> np.array:
        vector_1 = self.recommend_matrix[index_1].reshape(1, -1)
        vector_2 = self.recommend_matrix[index_2].reshape(1, -1)

        vector_1 = self._get_normalize(vector_1)
        vector_2 = self._get_normalize(vector_2)

        return vector_1 @ vector_2.T

    def _most_similar_object_by_index(
        self, index: int, topn: int = 10, exclude_self: bool = False
    ) -> pd.DataFrame:
        data_frame = self.construct_data_frame()

        recommend_matrix_norm = self._get_normalize(self.recommend_matrix)

        vector = recommend_matrix_norm[index]

        similar_vector = recommend_matrix_norm @ vector.T
        similar_vector = similar_vector.reshape(-1)

        if exclude_self:
            similar_vector[index] = -1

        rank_vector = similar_vector.argsort()[-topn:]
        similar_vector = similar_vector[rank_vector]

        similar_df = data_frame.loc[rank_vector, :]
        similar_df["similarity"] = similar_vector

        return similar_df.sort_values(["similarity"], ascending=False)

    def _find_index(self, name: str) -> int:
        """Because each recommendation system class has different motivation and different implementation,
        the base class cannot previously define `index` caller.
        The `_find_index()` function matches the item and their own item number (0 ~ ...).
        In general, the index has the form of calling `LabelEncoder` from `sklearn.preprocessing`.
        """
        return 0

    def _get_normalize(self, vector) -> np.array:
        """The function normalize vector into size 1.

        If vector is 2-dimensional array (matrix), then the function normalize each rows.
        """
        return self.normalizer.fit_transform(vector)