# Communitcation to boardgame site
import requests
from bs4 import BeautifulSoup

# Data Frame Module
import pandas as pd

# Useful Module
from tqdm.auto import tqdm

# Constant Module
from constants import *

class BoardgameMiner():
    """Get boardgame information from BGG (boardgamegeek.com)
    
    The crawler get the following informations from BGG
    1. Name
    2. Own Page
    3. Description
    4. Category
    5. Theme
    6. Mechanism
    ...

    If you want to crawl another information, inherit the class and modify `_crawl_single_boardgame()` function with `super()`
    
    The information is be used to the item-based recommendation system.
    """
    def __init__(self, page_max = PAGE_MAX, base_link = BASE_LINK, api_link = API_LINK, tracking = True):
        self.base_link = base_link
        self.api_link = api_link
        self.page_max = page_max
        self.tracking = tracking
        self.total_info = dict()

    def crawl_page(self):
        for page in self._track(range(1, self.page_max + 1), leave = True):
            self.total_info.update(self._crawl_single_page(page))
        
        return self.total_info

    def crawl_boardgame(self):
        info = self._track(self.total_info, leave = False)
        for boardgame in info:
            if type(info) != dict:
                info.set_postfix({'now': boardgame})

            simple_info = self.total_info[boardgame]
            uid = simple_info['uid']
            detail_info = self._crawl_single_boardgame(uid)
            self.total_info[boardgame].update(detail_info)
    
    def _crawl_single_page(self, page_num):
        req = requests.get(f"{self.base_link}/{page_num}")
        parser = BeautifulSoup(req.text, 'html.parser')
        
        boardgame_info = dict()
        
        for tag in self._track(parser.select(TABLE_TAG), leave = False, position = 0):
            name, description = self._crawl_single_page_row(tag)
            boardgame_info[name] = description
            
        return boardgame_info
    
    def _crawl_single_page_row(self, tag):
        info = tag.select('a')[2]

        name = info.text
        link = info.get('href')
        uid = link.split('/')[2]
        
        try:
            simple_description = tag.select_one('p').text.replace('\n', '').replace('\t', '')
        except:
            simple_description = ""
        
        return name, {'link': link, 'simple_description': simple_description, 'uid': uid}
    
    def _crawl_single_boardgame(self, uid):
        boardgame_info = requests.get(f"{self.api_link}/{uid}?{QUERY_STR}")
        bg_parser = BeautifulSoup(boardgame_info.text, 'html.parser')
        
        description = bg_parser.select_one(DESCRIPTION_TAG).text
        category = [tag.text for tag in bg_parser.select(CATEGORY_TAG)]
        mechanic = [tag.text for tag in bg_parser.select(MECHANIC_TAG)]
        theme = [tag.text for tag in bg_parser.select(THEME_TAG)]
        
        owner_num = bg_parser.select_one(OWNER_TAG).text

        return {'description': description, 'category': category, 'mechanic': mechanic, 'theme': theme, 'owner_num': owner_num}
    
    def _track(self, iterable_object, **kwarg):
        if self.tracking:
            return tqdm(iterable_object, **kwarg)
        return iterable_object

class BoardGameResultShower(BoardgameMiner):
    """The object shows the result of board game crawling in the pandas object."""
    def __init__(self, **kwarg):
        super().__init__(**kwarg)
    
    def get_data(self) -> pd.DataFrame:
        return pd.DataFrame(self.total_info)\
                 .transpose()