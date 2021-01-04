# pytorch framework
import torch
import torch.nn as NN
import torch.optim as Optim

# Useful Library
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Procedure Function
from tqdm.notebook import tqdm

class KorenALS():
    """Define Implicit Feedback Recommend System.
    The class is from "Collaborative filtering for implicit feedback datasets. (Hu, Koren, Volinsky)"
    
    However, the function wasn't work on the Macbook Pro 13' inch, 2019 late, because of the limitation of RAM
    """
    def __init__(self, sparse_data, user_column, item_column, value_column):
        """The sparse matrix has the form of
        | user_column | item_column |  value_column |
        |-------------|-------------|---------------|
        |     ...     |     ...     |      ...      |
        """
        self.user_column = user_column
        self.item_column = item_column
        self.value_column = value_column
        
        self.user_matcher = LabelEncoder().fit(sparse_data[user_column])
        self.category_matcher = LabelEncoder().fit(sparse_data[item_column])
        
        sparse_data[f'Encode_{user_column}'] = self.user_matcher.transform(sparse_data[user_column])
        sparse_data[f'Encode_{item_column}'] = self.category_matcher.transform(sparse_data[item_column])
        
        sparse_data = sparse_data.pivot(index = [f'Encode_{user_column}'], 
                                             columns = [f'Encode_{item_column}'],
                                             values = [value_column])[value_column].fillna(0)
        
        self.sparse_data = torch.Tensor(
            np.array(sparse_data)
        )
        self.user_num = len(sparse_data.index)
        self.item_num = len(sparse_data.columns)
    
    def training_model(self, alpha = 40, factor = 200, iteration = 10, regularization = 0.01):
        """Train the model with `ALS method`.
        
        In fact the loss function has the formula
        $\sum_{u, i}c_{ui}(\rho_{io}-x_u^Ty_i)^2 + \sum_{u}\|x_u\|^2 + \sum_{i}\|y_i\|^2$
        """
        self.train_info = {
            'alpha': alpha,
            'matrix-factor': factor,
            'iteration': iteration,
            'regularization': regularization
        }
        confidence = self.sparse_data * alpha + 1
        preference = (self.sparse_data > 0).float()
        
        U = torch.rand(self.user_num, factor)
        I = torch.rand(self.item_num, factor)
        
        for it in tqdm(range(iteration)):
            U = self._alternating_least_square(U, I, confidence, preference)
            I = self._alternating_least_square(I, U, confidence.transpose(0, 1), preference.transpose(0, 1))
        
        self.model = X @ Y
        return self.model
    
    def _alternating_least_square(self, variate_tensor, const_tensor, confidence, preference):
        """Calculate variate tensor so that reduces the objective function of Koren model.
        
        Parameter
        =========
        Variate_tensor = (K x N_f)
        Const_tensor   = (L x N_f)
        Confidence     = (K x L)
        Preference     = (K x L)
        """
        variate_tensor = torch.zeros_like(variate_tensor)
        variate_num = variate_tensor.size()[0]
        regularization = self.train_info['regularization']
        
        identity_matrix = torch.diag_embed(torch.zeros_like(variate_tensor[0]) + 1)
        const_transpose = const_tensor.transpose(0, 1)
        
        for index in range(variate_num):
            confidence_tensor = torch.diag_embed(confidence[index])
            preference_vector = preference[index]
            
            index_vector = const_transpose @ confidence_tensor @ const_tensor + regularization * identity_matrix
            index_vector = torch.inverse(index_vector) @ const_transpose @ confidence_tensor @ preference_vector
            
            variate_tensor[index] = index_vector
        
        return variate_tensor
    
    def report_model(self):
        self.model_df = pd.DataFrame(self.model)
        self.model_df.index = self.user_matcher.inverse_transform(self.model_df.index)
        self.model_df.column = self.user_matcher.inverse_transform(self.model_df.column)
        
        return self.model_df