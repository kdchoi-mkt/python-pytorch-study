# Type Hinting
import numpy as np
import pandas as pd


class BaseRS(object):
    def __init__(self, base_data_frame):
        self.base_data_frame = base_data_frame
        self.recommend_matrix = self.generate_recommend_matrix()

    def generate_recommend_matrix(self) -> np.array:
        """The function is for generate recommend matrix.
        The matrix can be as TF-IDF matrix, or N-dim embedding matrix.

        Anyway, each row represents the embedded vector of items.
        Note that the vector are not be normalized (i.e. $\\|x_i\\| \\not= 1$ in general.)

        If you inherit the object, you should modify the function.
        Of course, the return value is 2-dim array."""
        return np.zeros(1, 1)

    def construct_data_frame(self) -> pd.DataFrame:
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

    def _find_index(self, name: str) -> int:
        """Because each recommendation system class has different motivation and different implementation,
        the base class cannot previously define `index` caller (i.e. the class cannot call index from dataframe.index).
        The `_find_index()` function matches the item name and their own item number (0 ~ ...).
        In general, the index has the form of calling `LabelEncoder` from `sklearn.preprocessing`.
        """
        return 0
