from platforms import Platforms
from base_platform import BasePlatform
import requests
from endpoints import Endpoints
from request_types import RequestTypes
from datetime import datetime, timedelta

class Bitbucket(BasePlatform):
    def __init__(self, headers):
        self.headers = headers


    def fetch_page(self, url=None):
        date = (datetime.now() - timedelta(30)).strftime('%Y-%m-%d')
        params = {
            "after": date,
            "pagelen": 100, # Maximum allowed per page
        }
        if url is None:
            response = self.request_with_retry(Endpoints.BITBUCKET_SEARCH.value, RequestTypes.GET, headers=self.headers, params=params)
        else:
            response = self.request_with_retry(url, RequestTypes.GET, headers=self.headers)
        return response.json()


    def fetch_repositories(self, page_num, platform=""):
        repositories = []
        next_page = ""
        total_fetched = 0
        while total_fetched < page_num*100:  # 100 repos per page
            self.logger.info(f"Total of repositories fetched: {total_fetched}/{page_num*100}...")
            try:
                if next_page == "":
                    data = self.fetch_page()
                    next_page = data["next"]
                else:
                    data = self.fetch_page(url=next_page)
                    next_page = data["next"]
                # Filter and add only the values where the field "parent" is "null"
                repositories_without_forks = [repo for repo in data['values'] if repo.get("parent") is None]
                repositories.extend(repositories_without_forks)
                total_fetched += len(repositories_without_forks)
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Error while fetching repositories : {e}")
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
                    "default_branch": repo["mainbranch"]["name"] if repo.get("mainbranch") else None
                }
                for repo in repositories[:page_num*100]
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