# Parent class
from Crawling import BaseCrawling

# Communitcation to boardgame site
import requests
from bs4 import BeautifulSoup

# Data Frame Module
import pandas as pd

# Useful Module
from tqdm.auto import tqdm

# Constant Module
from BGConstants import *


class BoardgameMiner(BaseCrawling):
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

    def __init__(
        self,
        service=BASE_LINK,
        api=API_LINK,
        tracking=True,
        page_max=PAGE_MAX,
    ):
        super().__init__(
            service=service,
            api=api,
            tracking=tracking,
            page_max=page_max,
            query_str=QUERY_STR,
        )

    def _parse_service_response(self, parser):
        boardgame_info = dict()

        for tag in self._track(parser.select(TABLE_TAG), leave=False, position=0):
            name, description = self._parse_service_response_row(tag)
            boardgame_info[name] = description

        return boardgame_info

    def _parse_service_response_row(self, tag):
        info = tag.select("a")[2]

        name = info.text
        link = info.get("href")
        uid = link.split("/")[2]

        try:
            simple_description = (
                tag.select_one("p").text.replace("\n", "").replace("\t", "")
            )
        except:
            simple_description = ""

        return name, {
            "link": link,
            "simple_description": simple_description,
            "uid": uid,
        }

    def _parse_api_response(self, parser):
        description = parser.select_one(DESCRIPTION_TAG).text
        category = [tag.text for tag in parser.select(CATEGORY_TAG)]
        mechanic = [tag.text for tag in parser.select(MECHANIC_TAG)]
        theme = [tag.text for tag in parser.select(THEME_TAG)]

        owner_num = parser.select_one(OWNER_TAG).text

        return {
            "description": description,
            "category": category,
            "mechanic": mechanic,
            "theme": theme,
            "owner_num": owner_num,
        }


class BoardGameResultShower(BoardgameMiner):
    """DEPRECATED! get_data() function is inherited from BaseCrawling object.
    Use get_data_frame() for BoardgameMiner instead of get_data() from BoardgameResultShower.

    The object shows the result of board game crawling in the pandas object."""

    def __init__(self, **kwarg):
        super().__init__(**kwarg)

    def get_data(self) -> pd.DataFrame:
        return pd.DataFrame(self.total_info).transpose()
