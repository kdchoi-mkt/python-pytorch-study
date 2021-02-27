from sklearn.feature_extraction.text import TfidfVectorizer

# Item Description Based Recommendation System
class DescriptionBasedRS(TfidfVectorizer):
    def __init__(self, data_frame, description_column, name_column, **kwarg):
        super().__init__(**kwarg)
        self.data_frame = data_frame
        self.name_col = name_column
        self.description_col = description_column
        self.vectorized_form = self.fit_transform(data_frame[description_column])

    def similarity_two_object(self, name_1, name_2):
        index_1 = self._find_index(name_1)
        index_2 = self._find_index(name_2)

        return self._similarity_two_object_by_index(index_1, index_2)
    
    def most_similar_object(self, name, topn = 10, exclude_self = False):
        index = self._find_index(name)

        return self._most_similar_object_by_index(index, topn, exclude_self)

    def _similarity_two_object_by_index(self, index_1, index_2):
        vector_1 = self.vectorized_form[index_1].toarray()
        vector_2 = self.vectorized_form[index_2].toarray()
        
        return vector_1 @ vector_2.T

    def _most_similar_object_by_index(self, index, topn = 10, exclude_self = False):
        data_frame = self.data_frame.reset_index(drop = True)
        vector = self.vectorized_form[index].toarray()
        
        similar_vector = self.vectorized_form @ vector.T
        similar_vector = similar_vector.reshape(-1)

        if exclude_self:
            similar_vector[index] = -1

        rank_vector = similar_vector.argsort()[-topn:]
        similar_vector = similar_vector[rank_vector]

        similar_df = data_frame.iloc[rank_vector, :]
        similar_df['similarity'] = similar_vector
        
        return similar_df.sort_values(['similarity'], ascending = False)

    def _find_index(self, name):
        if name not in self.data_frame[self.name_col].to_list():
            raise ValueError(f"The boardgame name {name} does not in the database. Try again!")

        target_data = self.data_frame.query(f"{self.name_col} == '{name}'")
        return target_data.index[0]