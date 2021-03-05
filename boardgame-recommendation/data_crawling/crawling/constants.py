PAGE_MAX = 5

# These Consts are used in BoardgameMiner
# BG: Boardgame

BG_BASE_LINK = "https://boardgamegeek.com/browse/boardgame/page"
BG_TABLE_TAG = "#row_"

BG_API_LINK = "https://boardgamegeek.com/xmlapi/boardgame/"
DESCRIPTION_TAG = "description"
MECHANIC_TAG = "boardgamemechanic"
CATEGORY_TAG = "boardgamecategory"
THEME_TAG = "boardgamesubdomain"

BG_QUERY_STR = "stats=1"
OWNER_TAG = "owned"

# These Consts are used in UserMiner
# OG: Own Game

REVIEW_LINK = "https://boardgamegeek.com/thread/browse/boardgame/0/page"
REVIEW_TABLE_TAG = "table > tr"
USER_NAME_TAG = "td > div > span > a"

# These Consts are only used in UserMiner

OG_API_LINK = "https://www.boardgamegeek.com/xmlapi/collection"
ITEM_TAG = "item"
BOARDGAME_TAG = "name"
COMMENT_TAG = "comment"
RATING_TAG = "rating"

OG_QUERY_STR = ""
COMMENT_TAG = "comment"

# These Consts are used in PlayHistoryMiner
# PH: Play History

PH_API_LINK = "https://www.boardgamegeek.com/xmlapi2/plays"
PH_QUERY_STR_FTN = lambda user_name: f"username={user_name}"
PLAY_TAG = "play"
BOARDGAME_NAME_TAG = "item"
