from platforms import Platforms
from base_platform import BasePlatform
import requests
from endpoints import Endpoints
from request_types import RequestTypes
from metrics import Metrics
from datetime import datetime, timedelta

class Bitbucket(BasePlatform):
    """
    This class defines the Bitbucket platform-specific matcher.
    """
    def __init__(self, headers):
        self.headers = headers


    def fetch_page(self, url=None):
        """
        Function to fetch a page of repositories from the GitHub API.
        :param page: The page number to fetch.
        :param url: The URL to fetch.
        :return: JSON response from the API.
        """
        yesterday = (datetime.now() - timedelta(1)).strftime('%Y-%m-%d')
        params = {
            "after": yesterday,
            "pagelen": 100, # Maximum allowed per page
        }
        if url is None:
            response = self.request_with_retry(Endpoints.BITBUCKET_SEARCH.value, RequestTypes.GET, headers=self.headers, params=params)
        else:
            response = self.request_with_retry(url, RequestTypes.GET, headers=self.headers)
        return response.json()


    def fetch_repositories(self, page_num, platform=""):
        """
        Function to fetch a given number of pages of repositories from the GitHub API.
        :param page_num: Number of pages to fetch.
        :param platform: Platform to fetch repositories from.
        :return: List of dictionaries containing repository data.
        """
        repositories = []
        next_page = ""
        for page in range(1, page_num):  # 100 repos per page
            self.logger.info(f"Fetching page {page}...")
            try:
                if next_page == "":
                    data = self.fetch_page()
                    next_page = data["next"]
                else:
                    data = self.fetch_page(url=next_page)
                    next_page = data["next"]
                repositories.extend(data['values'])
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Error fetching page {page}: {e}")
                break

        # Extract relevant repository data
        if repositories:
            data = [
                {
                    "platform": "Bitbucket",
                    "owner": repo["workspace"]["slug"],
                    "repo": repo["slug"],
                    "id": repo["uuid"],
                    "created": repo["created_on"],
                    "updated": repo["updated_on"],
                    "default_branch": repo["mainbranch"]["name"]
                }
                for repo in repositories
            ]
            return data
        return None

    def get_metric(self,metric,owner,repo,platform=Platforms.BITBUCKET):

        temp = platform.name + "_" + metric.name
        url = getattr(Endpoints, temp)(owner, repo)

        try:
            response = self.request_with_retry(url, RequestTypes.GET, headers=self.headers)
            if not str(response.status_code).startswith('4'):
                data = response.json()
                return data.get("size", None)
            else:
                self.logger.error(f"Error fetching {owner}/{repo}: {response.status_code}")
                return None
        except Exception as e:
            self.logger.error(f"Exception fetching {owner}/{repo}: {e}")
            return None