from abc import ABC, abstractmethod
import requests
from endpoints import Endpoints
import sys
import logging
from tenacity import retry, stop_after_attempt, retry_if_exception_type, wait_fixed, before_log,after_log
from requests.exceptions import HTTPError, Timeout
from metrics import Metrics
from request_types import RequestTypes


class BasePlatform(ABC):
    """
    This file defines the abstract base class for platform-specific matchers.
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger(__name__)


    @abstractmethod
    def fetch_repositories(self, page_num, platform):
        """
        Function to fetch a given number of pages of repositories from a platform-specific API.
        :param page_num: Number of pages to fetch.
        :param platform: Platform to fetch repositories from.
        """
        pass


    @retry(stop=stop_after_attempt(4), wait=wait_fixed(30),
           retry=retry_if_exception_type((HTTPError, Timeout)),
           before=before_log(logger, logging.INFO), after=after_log(logger, logging.INFO))
    def request_with_retry(self, url, request_type, headers=None, params=None):
        """
        Function to make a request to a given URL with retry logic based on Tenacity.
        :param url: URL to make the request to.
        :param request_type: Type of request to make (GET or POST).
        :param headers: Headers to include in the request.
        :param params: Parameters to include in the request
        :return: Response from the request
        """
        # Check Request Type (post request used for GraphQL api) and Make Request
        if request_type == RequestTypes.GET:
            response = requests.get(url, headers=headers, params=params)
        elif request_type == RequestTypes.POST:
            response = requests.post(url, headers=headers, json=params)

        # Check Specific Response Status Code to avoid useless retries
        if response.status_code == 404:
            self.logger.error(f"Error fetching {url}: not found (404)")
            return response
        if response.status_code == 409:
            self.logger.error(f"Error fetching {url}: empty repository")
            return response
        if response.status_code == 500:
            self.logger.error(f"Error fetching {url}: internal server error (500)")
            return response
        response.raise_for_status()
        return response


    def get_metric(self,platform,metric,owner,repo):
        """
        Function to fetch a given metric from a platform-specific API.
        :param platform: Platform to fetch the metric from.
        :param metric: Metric to fetch.
        :param owner: Owner of the repository.
        :param repo: Repository name.
        :return: Value of the metric for the given repository.
        """
        temp = platform.name + "_" + metric.name
        url = getattr(Endpoints, temp)(owner, repo)

        try:
            response = self.request_with_retry(url, headers=self.headers)
            if not str(response.status_code).startswith('4'):
                link_header = response.headers.get("Link", "")
                if 'rel="last"' in link_header:
                    last_page = link_header.split("page=")[-1].split(">")[0].split("&")[0]
                    return int(last_page)
                elif metric == Metrics.ISSUE or metric == Metrics.PULL_REQUEST:
                    return 0  # If no pagination, assume 0 entries
                else:
                    return 1  # If no pagination, assume 1 entry
            else:
                self.logger.error(f"Error fetching {owner}/{repo}: {response.status_code}")
                return None
        except Exception as e:
            self.logger.error(f"Exception fetching {owner}/{repo}: {e}")
            return None

    def add_metric(self, df, platform, metric):
        """
        Function to add a given metric to a DataFrame of repositories.
        :param df: DataFrame of repositories.
        :param platform: Platform to fetch the metric from.
        :param metric: Metric to fetch.
        :return: DataFrame with the added metric column.
        """
        metric_counts = []
        for index, row in df.iterrows():
            owner, repo = row["owner"], row["repo"]
            self.logger.info(f"Fetching data for {owner}/{repo}...")
            metric_counts.append(self.get_metric(platform,metric,owner,repo))

        df[metric.value] = metric_counts
