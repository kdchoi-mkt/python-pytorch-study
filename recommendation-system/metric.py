import pandas as pd

def precision_k(train_data, test_data, k, user_col, item_col, rank_col = 'rank'):
    """Derive Precision@K for the recommendation system
    
    Train, and test data has the stacked form, and the value is the ranking.
    The range of ranking is [1, N] and the ranking is descending order.
    
    Precision = Actual K / Recommend K
    Ex)
    | user_id | item_id | rank |
    |---------|---------|------|
    | 121000  |   012   |   3  |
    |  ....   |   ...   |  ... |    
    """
    train_data = train_data[(train_data[rank_col] <= k) & (train_data[user_col].isin(test_data[user_col]))]
    test_data = test_data[test_data[rank_col] <= k]
    
    precision = pd.merge(train_data, test_data, on = [user_col, item_col], how = 'left', suffixes = ('_train', '_test'))
    precision_gp = precision.groupby([user_col])

    precision_df = pd.DataFrame(
        data = [
            precision_gp[f'{rank_col}_test'].count(),
            precision_gp[f'{rank_col}_train'].count()
        ],
        index = [
            'actual_liked',
            'recommended'
        ]
    ).transpose().reset_index()
    precision_df['precision@k'] = precision_df['actual_liked'] / precision_df['recommended']
    
    return precision_df['precision@k'].mean()

def recall_k(train_data, test_data, k, user_col, item_col, rank_col = 'rank'):
    """Derive Precision@K for the recommendation system
    
    Train, and test data has the stacked form, and the value is the ranking.
    The range of ranking is [1, N] and the ranking is descending order.
    
    Recall = Recommend K / Actual K
    Ex)
    | user_id | item_id | rank |
    |---------|---------|------|
    | 121000  |   012   |   3  |
    |  ....   |   ...   |  ... |    
    """
    train_data = train_data[(train_data[rank_col] <= k) & (train_data[user_col].isin(test_data[user_col]))]
    test_data = test_data[(test_data[rank_col] <= k) & (test_data[user_col].isin(train_data[user_col]))]
    
    recall = pd.merge(test_data, train_data, on = [user_col, item_col], how = 'left', suffixes = ('_test', '_train'))
    recall_gp = recall.groupby([user_col])

    recall_df = pd.DataFrame(
        data = [
            recall_gp[f'{rank_col}_test'].count(),
            recall_gp[f'{rank_col}_train'].count()
        ],
        index = [
            'actual_liked',
            'recommended'
        ]
    ).transpose().reset_index()
    recall_df['recall@k'] = recall_df['recommended'] / recall_df['actual_liked']
    
    return recall_df['recall@k'].mean()

def expected_rank_percentile(train_data, test_data, user_col, item_col, rank_col = 'rank'):
    """Derive ERP for the recommendation system.
    The measurement is from "Hu, Koren, Volinsky (2008)"
    
    Train, and test data has the stacked form, and the value is the ranking.
    The range of ranking is [1, N] and the ranking is descending order.
    
    Ex)
    | user_id | item_id | rank |
    |---------|---------|------|
    | 121000  |   012   |   3  |
    |  ....   |   ...   |  ... |  
    """
    train_data = train_data[train_data[user_col].isin(test_data[user_col])]
    test_data = test_data[test_data[user_col].isin(train_data[user_col])]
    
    valid_df = pd.merge(train_data, test_data, on = [user_col, item_col], how = 'left', suffixes = ('_train', '_test'))
    valid_df['rank_percentile'] = (valid_df[f'{rank_col}_train'] - 1) / valid_df[f'{rank_col}_train'].max() * 100
    return valid_df[~pd.isna(valid_df[f'{rank_col}_test'])]['rank_percentile'].mean()

def average_relative_position(train_data, test_data, user_col, item_col, rank_col = 'rank'):
    """Derive ARP for the recommendation system.
    The measurement is from "Fast ALS-based Matrix Factorization for Explicit and Implicit Feedback Datasets"
    
    Train, and test data has the stacked form, and the value is the ranking.
    The range of ranking is [1, N] and the ranking is descending order.
    
    Ex)
    | user_id | item_id | rank |
    |---------|---------|------|
    | 121000  |   012   |   3  |
    |  ....   |   ...   |  ... |
    """
    train_data = train_data[train_data[user_col].isin(test_data[user_col])]
    test_data = test_data[test_data[user_col].isin(train_data[user_col])]
    
    valid_df = pd.merge(train_data, test_data, on = [user_col, item_col], how = 'left', suffixes = ('_train', '_test'))

    valid_df = valid_df.sort_values([f'{user_col}', f'{rank_col}_train'])

    valid_df['implicit_exist'] = ~pd.isna(valid_df[f'{rank_col}_test']) * 1

    valid_df['cum_exist'] = valid_df.groupby(user_col)['implicit_exist'].cumsum()

    valid_df['revealed_item'] = valid_df.groupby(user_col)[f'{rank_col}_test'].transform(func = 'count')

    valid_df['relative_position'] = valid_df['rank_train'] - valid_df['cum_exist']
    valid_df['relative_position'] = valid_df['relative_position'] / (valid_df[f'{rank_col}_train'].max() - valid_df['revealed_item']) * 100

    revealed_df = valid_df[valid_df['implicit_exist'] == 1]

    return revealed_df['relative_position'].mean()