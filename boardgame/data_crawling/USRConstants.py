PAGE_MAX = 5

BASE_LINK = "https://boardgamegeek.com/thread/browse/boardgame/0/page"
TABLE_TAG = "table > tr"
USER_TAG = "td > div > span > a"

OWN_GAME_API_LINK = "https://www.boardgamegeek.com/xmlapi/collection"
ITEM_TAG = "item"
BOARDGAME_TAG = "name"
COMMENT_TAG = "comment"
RATING_TAG = "rating"

QUERY_STR = ""
COMMENT_TAG = "comment"

# These Consts are used in PlayHistoryMiner

PLAY_HISTORY_API_LINK = "https://www.boardgamegeek.com/xmlapi2/plays"
QUERY_STR_FTN = lambda user_name: f"username={user_name}"
PLAY_TAG = "play"
BOARDGAME_NAME_TAG = "item"
