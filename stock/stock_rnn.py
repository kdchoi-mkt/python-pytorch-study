import torch
import torch.nn as NN
import random

# Define RNN cell for learning stock data.
class StockRNN(NN.Module):
    """RNN(recurrence nerual network) implement for stock data
    
    The single RNN cell runs the following steps:
    1. Combine input_data and hidden_data from previous cell, call concat_data
    2. The concat data goes through two distinct functions
        `i2o`: concat_data to output
        `i2h`: concat_data to hidden
    3. Finally, the RNN cell reports the output prediction and hidden states.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.i2o = NN.Sequential(
            NN.Linear(input_size + hidden_size, 128),
            NN.ReLU(),
            NN.Linear(128, output_size),
        )
        self.i2h = NN.Sequential(
            NN.Linear(input_size + hidden_size, hidden_size),
            NN.Softmax()
        )
    
    def forward(self, input, hidden):
        cat = torch.cat([input, hidden], dim = 1)
        return self.i2o(cat), self.i2h(cat)
    
    def initHidden(self):
        return torch.zeros(1, self.hidden_size)
    
def choice_random_train(stock_dict: dict, stock_list: list):
    """Choice random element from stock dictionary.
    
    In fact, for any dictionary that has the value of JSON {'input': ~~, 'output': ~~}, the function could apply.
    """
    
    stock_name = random.choice(stock_list)
    stock_data = stock_dict[stock_name]
    
    input_data = stock_data['input']
    output_data = stock_data['output']
    
    random_index = random.randint(0, len(input_data) - 1)
    
    return {
        'input': input_data[random_index].unsqueeze(1), 
        'output': output_data[random_index][1].unsqueeze(0).unsqueeze(0), # 마지막은 volume. 거래량 예측은 필요 없음
        'stock_name': stock_name
    }
    

def evaluate_time_series(stock_info, rnn_model):
    """Evaluate time series with RNN"""
    
    hidden = rnn_model.initHidden()
    
    for stock_t in stock_info['input']:
        output, hidden = rnn_model(stock_t, hidden)
        
    return output
