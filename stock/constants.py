CODE_CRAWL_LINK = "http://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13"
FINANCE_CRAWL_LINK = lambda CODE, PAGE: f"http://finance.naver.com/item/sise_day.nhn?code={CODE}&page={PAGE}"
STOCK_VALUE_LIST = ['시가', '종가', '저가', '고가']