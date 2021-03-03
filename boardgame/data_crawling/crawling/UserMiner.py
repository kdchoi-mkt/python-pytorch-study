# Parent class
from data_crawling.base import BaseMiner

# Constant Module
from data_crawling.crawling.constants import *


class UserMiner(BaseMiner):
    """Get BoardgameGeek user information from boardgamegeek.com.

    The crawler get the following informations from BGG
    1. User Name
    2. Own Game
    3. Commented Game

    Contrasts to the boardgame information, the user information is limited to gather in sort of their popularity, else.
    Therefore, I gather the information through the recent reviewing activity on the boardgame geek community."""

    def __init__(
        self,
        service=REVIEW_LINK,
        api=OG_API_LINK,
        tracking=True,
        page_max=PAGE_MAX,
        query_str=OG_QUERY_STR,
    ):
        super().__init__(
            service=service,
            api=api,
            tracking=tracking,
            page_max=page_max,
            query_str=query_str,
        )

    def _parse_service_response(self, parser):
        """The function does not collect any other information than user"""
        user_info = dict()

        for tag in parser.select(REVIEW_TABLE_TAG):
            try:
                name = tag.select(USER_NAME_TAG)[0].text
                user_info[name] = {"uid": name}
            except:
                continue

        return user_info

    def _parse_api_response(self, parser):
        """The function collects the own boardgame and commented boardgame.
        If the comment does not exist, the comment would be ""
        ```
        User 1: {
            Gloomhaven: {
                comment: "Gloomhaven is God Thing Boardgame!!",
                rating: 4
            },
            Dominion: {
                comment: "",
                rating: 10,
            }
        }
        ```
        """
        owned_item = list()

        for tag in parser.select(ITEM_TAG):
            item_info = dict()
            item = tag.select_one(BOARDGAME_TAG).text

            rating = self._error_treat(
                lambda x: tag.select_one(x).get("value"), RATING_TAG, "N/A"
            )
            comment = self._error_treat(
                lambda x: tag.select_one(x).text, COMMENT_TAG, ""
            )

            item_info[item] = {"rating": rating, "comment": comment}

            owned_item.append(item_info)

        return {"owned_item": owned_item}

    def to_data_frame(self):
        return self.to_data_frame_preprocess()

    def to_data_frame_preprocess(self, preprocess_col="owned_item"):
        """The UserMiner class supports automatical preprocessing stage"""
        data_frame = super().to_data_frame()
        data_frame = data_frame.explode(preprocess_col).dropna()

        data_frame["boardgame"] = data_frame[preprocess_col].apply(
            lambda json: list(json.keys())[0]
        )
        data_frame["values"] = data_frame[preprocess_col].apply(
            lambda json: list(json.values())[0]
        )
        return data_frame


class PlayHistoryMiner(UserMiner):
    """Get User's Play History from boardgamegeek.com.

    Because the gathering initial user info is same as UserMiner, the object inherits `UserMiner` object.
    If we can collect the play history, we can embed the boardgame into the n-dim vector by item 2 vec.
    """

    def __init__(
        self,
        service=REVIEW_LINK,
        api=PH_API_LINK,
        tracking=True,
        page_max=PAGE_MAX,
        query_str_ftn=PH_QUERY_STR_FTN,
    ):
        super().__init__(
            service=service,
            api=api,
            tracking=tracking,
            page_max=page_max,
            query_str="",
        )
        self.api_combination = lambda uid: f"{self.api}?{query_str_ftn(uid)}"

    def _parse_api_response(self, parser):
        """The function collects the users' previous game history. The limitation of the API, I can gather only 100 recent records per user.
        ```
        User 1: {
            play_history: [
                {
                    'Gloomhaven': '2011-01-03'
                },
                {
                    'Dominion': '2011-01-04'
                },
                ...
            ]
        }
        ```
        """
        play_history = list()

        for tag in parser.select(PLAY_TAG):
            segment = dict()

            item = tag.select_one(BOARDGAME_NAME_TAG).get("name")
            played_at = self._error_treat(lambda x: tag.get(x), "date", "1969-12-31")

            segment[item] = {"played_at": played_at}

            play_history.append(segment)

        return {"play_history": play_history}

    def to_data_frame_preprocess(self):
        return super().to_data_frame_preprocess("play_history")
