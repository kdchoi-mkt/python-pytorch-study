# Parent class
from Crawling import BaseCrawling

# Constant Module
from constants import *


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

    The information is be used to the item-based recommendation system.
    """

    def __init__(
        self,
        service=BG_BASE_LINK,
        api=BG_API_LINK,
        tracking=True,
        page_max=PAGE_MAX,
    ):
        super().__init__(
            service=service,
            api=api,
            tracking=tracking,
            page_max=page_max,
            query_str=BG_QUERY_STR,
        )

    def _parse_service_response(self, parser):
        boardgame_info = dict()

        for tag in self._track(parser.select(BG_TABLE_TAG), leave=False, position=0):
            name, description = self._parse_service_response_row(tag)
            boardgame_info[name] = description

        return boardgame_info

    def _parse_service_response_row(self, tag):
        info = tag.select("a")[2]

        name = info.text
        uid = info.get("href").split("/")[2]

        try:
            simple_description = (
                tag.select_one("p").text.replace("\n", "").replace("\t", "")
            )
        except:
            simple_description = ""

        return name, {
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
