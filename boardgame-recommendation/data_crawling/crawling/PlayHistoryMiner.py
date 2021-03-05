# Parent class
from .UserMiner import UserMiner

# Constant Module
from .constants import *


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
