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
from USRConstants import *


class BoardgameUserMiner(BaseCrawling):
    """Get BoardgameGeek user information from boardgamegeek.com.

    Contrasts to the boardgame information, the user information is limited to gather as many as possible in sort of their popularity, else.
    Therefore, I gather the information with the recent reviewing activity on the boardgame geek community."""

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
        """The function does not collect any other information than user"""
        user_info = dict()

        for tag in self._track(parser.select(TABLE_TAG)[1:], leave=False):
            name = tag.select(USER_TAG)[0].text
            user_info[name] = {"uid": name}

        return user_info

    def _parse_api_response(self, parser):
        """The function collects the own boardgame and commented boardgame.
        If the comment does not exist, the comment would be ""
        ```
        User 1: {
            Gloomhaven: {
                comment: "~~~~~~"
            },
            Dominion: {
                comment: ""
            }
        }
        ```
        """

        owned_item = [tag.text for tag in parser.select(BOARDGAME_TAG)]
        commented_item = [
            tag.select_one(BOARDGAME_TAG).text
            for tag in parser.select(ITEM_TAG)
            if tag.select_one(COMMENT_TAG) != None
        ]

        return {"own_boardgame": owned_item, "commented_boardgame": commented_item}
