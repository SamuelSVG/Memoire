from abc import ABC, abstractmethod
import requests
from endpoints import Endpoints
import sys
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from requests.exceptions import HTTPError, Timeout


class BasePlatform(ABC):
    """
    Abstract base class that defines the interface for platform-specific matchers.
    """

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger(__name__)

    @abstractmethod
    def fetch_repositories(self, page_num, platform):
        """Load the platform's data"""
        pass

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=30, max=60),
           retry=retry_if_exception_type((HTTPError, Timeout)))
    def request_with_retry(self, url, headers=None, params=None):
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response

    # Function to Fetch Metrics from GitHub API
    def get_metric(self,platform,metric,owner,repo):
        temp = platform.name + "_" + metric.name
        url = getattr(Endpoints, temp)(owner, repo)

        try:
            response = self.request_with_retry(url, headers=self.headers)
            if response.status_code == 200:
                link_header = response.headers.get("Link", "")
                if 'rel="last"' in link_header:
                    last_page = link_header.split("page=")[-1].split(">")[0]
                    return int(last_page)
                return 1  # If no pagination, assume 1 entry
            else:
                self.logger.error(f"Error fetching {owner}/{repo}: {response.status_code}")
                return None
        except Exception as e:
            self.logger.error(f"Exception fetching {owner}/{repo}: {e}")
            return None

    def add_metric(self, df, platform, metric):
        # Iterate Over Each Repository and Fetch Metrics
        metric_counts = []
        for index, row in df.iterrows():
            owner, repo = row["owner"], row["repo"]
            self.logger.info(f"Fetching data for {owner}/{repo}...")
            metric_counts.append(self.get_metric(platform,metric,owner,repo))

        # Add Results to DataFrame
        df[metric.value] = metric_counts
