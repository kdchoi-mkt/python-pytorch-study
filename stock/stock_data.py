import torch
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm.notebook import tqdm
from constants import CODE_CRAWL_LINK, FINANCE_CRAWL_LINK, STOCK_VALUE_LIST

class StockData:
    """주식 거래 데이터 중 시가 | 종가 | 고가 | 저가 4개의 데이터를 네이버 금융을 통해 가져오는 클래스입니다.
    
    Pandas DataFrame을 통해서 전처리가 이루어지며, pyTorch를 통한 텐서로의 변환 역시 가능합니다.
    """
    def __init__(self, target_code = None, reference_data = None):
        self.stock_data = self.get_stock_data(target_code)

    def get_stock_data(self, target_code) -> pd.DataFrame:
        self.code_list = self._get_code_list(target_code)
        self.stock_data = self._get_stock_data_by_code_list(self.code_list)

        return self.stock_data

    def _get_code_list(self, target_code) -> list:
        if target_code != None:
            return [target_code]

        code_info = pd.read_html(CODE_CRAWL_LINK)[0]
        code_list = code_info['종목코드'].apply(lambda x: ''.join(['0' for _ in range(6 - len(str(x)))]) + str(x))

        return code_list

    def _get_stock_data_by_code_list(self, code_list: list) -> pd.DataFrame():
        stock_data = pd.DataFrame()
        
        for code in tqdm(code_list):
            code_stock_data = self._get_stock_data_by_code(code)
            stock_data = pd.concat([stock_data, code_stock_data])

        return stock_data

    def _get_stock_data_by_code(self, code: str) -> pd.DataFrame:
        code_data = pd.DataFrame()
        for page in range(40):
            try:
                code_data = pd.concat([code_data, self._get_stock_data_by_code_single(code, page)])
            except:
                pass

        if len(code_data) == 0:
            return code_data

        code_data = code_data.sort_values('날짜', ascending = True)
        return code_data

    def _get_stock_data_by_code_single(self, code, page) -> pd.DataFrame:
        local_code_info = pd.read_html(FINANCE_CRAWL_LINK(code, page))[0]
        local_code_info = local_code_info.dropna(axis = 0)

        local_code_info['코드'] = code
        local_code_info['날짜'] = local_code_info['날짜'].apply(lambda x: datetime.strptime(x, '%Y.%m.%d'))

        return local_code_info

'''
# StockData.to_tensor is Deprecated.

    def to_tensor(self) -> torch.Tensor:
        """Stock_data를 Tensor로 변경합니다.
        기존의 stock data 변수는 그대로 남아있으며, 이 함수를 실행시킬 시 stock_data_tensor 데이터가 만들어집니다.

        Tensor는 기본적으로 C x N x 4의 dimension을 가집니다. C는 코드의 개수이고, N은 각 코드가 가지고 있는 데이터의 개수입니다.
        코드 순서는 code_list와 같습니다.
        각 데이터는 시가 (Open) | 종가 (Close) | 저점 (Low) | 고점 (High) 로 이루어져있습니다.
        """ 

        refined_dataframe = self.stock_data[['코드'] + STOCK_VALUE_LIST]
        code_list = list()
        self.stock_data_tensor = torch.Tensor()

        for code in self.code_list:
            local_tensor = refined_dataframe[refined_dataframe['코드'] == code][STOCK_VALUE_LIST]
            if len(local_tensor) > 0:
                code_list.append(code)
            local_tensor = torch.Tensor([np.array(local_tensor)])
            self.stock_data_tensor = torch.cat([self.stock_data_tensor, local_tensor])

        self.code_list = code_list
        return self.stock_data_tensor
'''