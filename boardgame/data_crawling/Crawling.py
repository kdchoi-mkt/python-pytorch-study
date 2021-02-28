from tqdm.auto import tqdm
from bs4 import BeautifulSoup
import requests
import pandas as pd


class BaseCrawling:
    """The basic crawling object, especially the abstract interface.

    The `BasicCrawling` object would be extended using BoardgameMiner, UserMiner, etc.
    In fact, the crawling runs by the following mechanisms:

    1) Gather basic information from SERVICE
        + The SERVICE link must have the form: {MAIN_LINK}/{PAGE_NUM}
    2) Gather specific information from API

    If you want, you can track whether the running is well-done by `tqdm` module.
    The object have the private function `_track() -> Union[tqdm, iteration]` so that you can easily implement.
    """

    def __init__(
        self, service: str, api: str, query_str: str, tracking: bool, page_max: int
    ):
        self.service = service
        self.api = api
        self.tracking = tracking
        self.page_max = page_max
        self.query_str = query_str
        self.total_info = dict()

    def crawl_page(self):
        """Gather the information from the service.
        The function executes `_crawl_single_page()` locally.
        """
        for page in self._track(range(1, self.page_max + 1), leave=True):
            self.total_info.update(self._crawl_single_page(page))

        return self.total_info

    def gather_detail_info(self):
        """Gather detail information from the service.
        If you wants to run the function, you must execute `crawl_page()` before.

        Also, you **MUST** gather `uid` in `crawl_page()` stage, too.
        """
        info = self._track(self.total_info, leave=False)

        for item in info:
            if type(item) != dict:
                info.set_postfix({"now": item})

            uid = self.total_info[item]["uid"]
            detail_info = self._call_api(uid)

            self.total_info[item].update(detail_info)

    def _crawl_single_page(self, page_num) -> dict:
        """Gather the information from the services' page."""
        req = requests.get(f"{self.service}/{page_num}")
        parser = BeautifulSoup(req.text, "html.parser")

        information = self._parse_service_response(parser)

        return information

    def _call_api(self, uid) -> dict:
        """Gather the information from the API.
        If you inherit the object, you should modify this function.
        """
        req = requests.get(f"{self.api}/{uid}?{self.query_str}")
        parser = BeautifulSoup(req.text, "html.parser")

        information = self._parse_api_response(parser)

        return information

    def _parse_service_response(self, parser) -> dict:
        """If you inherit the object, you should modify this function."""
        return dict()

    def _parse_api_response(self, parser) -> dict:
        """If you inherit the object, you should modify this function."""
        return dict()

    def _track(self, iterable_object, **kwarg):
        if self.tracking:
            return tqdm(iterable_object, **kwarg)
        return iterable_object

    def to_data_frame(self):
        return pd.DataFrame(self.total_info).transpose()