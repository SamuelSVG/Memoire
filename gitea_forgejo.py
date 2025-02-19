# Headers for different platforms
from base_platform import BasePlatform
import time
import requests
import pandas as pd
from endpoints import Endpoints
from metrics import Metrics

class GiteaForgejo(BasePlatform):
    def __init__(self, headers):
        self.headers = headers

    def fetch_page(self, page, platform):
        params = {
            "q": "",  # Empty query to fetch all repositories
            "sort": "updated",  # Sort by most recently updated
            "order": "desc",  # Descending order
            "limit": 50,  # Maximum allowed per page
            "page": page,  # Current page
        }
        platform_endpoint = platform.name + "_SEARCH"
        url = getattr(Endpoints, platform_endpoint).value
        response = self.request_with_retry(url, headers=self.headers, params=params)
        return response.json()

    def fetch_repositories(self, page_num, platform=""):
        repositories = []
        for page in range(1, page_num):  # 10 pages × 100 repos per page = 1,000 repos
            self.logger.info(f"Fetching page {page}...")
            try:
                data = self.fetch_page(page, platform)
                repositories.extend(data.get("data", []))  # "data" holds the repo list
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Error fetching page {page}: {e}")
                break

        if repositories:
            data = [
                {
                    "platform": platform.value.capitalize(),
                    "owner": repo["owner"]["username"],
                    "repo": repo["name"],
                    "id": repo["id"],
                    "created": repo["created_at"],
                    "updated": repo["updated_at"],
                    "language": repo.get("language", None),
                    "license": repo.get("licenses", [None])[0] if repo.get("licenses") else None,
                    "size": repo["size"],
                    "#stars": repo["stars_count"],
                    "#forks": repo["forks_count"]
                }
                for repo in repositories
            ]
            return data
        return None

    """
    # Function to fetch repositories
    def fetch_repositories(self, page_num, platform):
        repositories = []
        for page in range(1, page_num):  # 20 pages × 50 repos per page = 1,000 repos
            self.logger.info(f"Fetching page {page}...")
            params = {
                "q": "",  # Empty query to fetch all repositories
                "sort": "updated",  # Sort by most recently updated
                "order": "desc",  # Descending order
                "limit": 50,  # Maximum allowed per page
                "page": page,  # Current page
            }
            platform_endpoint = platform.name + "_SEARCH"
            url = getattr(Endpoints, platform_endpoint).value
            response = requests.get(url, headers=self.headers, params=params)

            # Check for errors
            if response.status_code != 200:
                self.logger.error(f"Error: {response.status_code}, {response.json()}")
                break

            data = response.json()
            repositories.extend(data.get("data", []))  # "data" holds the repo list
            time.sleep(5)  # Reduce load on the API

        if repositories:
            data = [
                {
                    "platform": platform.value.capitalize(),
                    "owner": repo["owner"]["username"],
                    "repo": repo["name"],
                    "id": repo["id"],
                    "created": repo["created_at"],
                    "updated": repo["updated_at"],
                    "language": repo.get("language", None),
                    "license": repo.get("licenses", [None])[0] if repo.get("licenses") else None,
                    "size": repo["size"],
                    "#stars": repo["stars_count"],
                    "#forks": repo["forks_count"]
                }
                for repo in repositories
            ]
            return data
        return None
    """

    # Function to Fetch #collaborators metric from Gitea/Forgejo API
    def get_contributors(self, platform, owner, repo):
        url = getattr(Endpoints, platform.name + "_CONTRIBUTOR")(owner, repo)
        try:
            response = self.request_with_retry(url, headers=self.headers)
            if response.status_code == 200:
                return len(response.json())-1
            else:
                self.logger.error(f"Error fetching {owner}/{repo}: {response.status_code}")
                return None
        except Exception as e:
            self.logger.exception(f"Exception fetching {owner}/{repo}: {e}")
            return None

    def add_contributors(self, df, platform):
        # Iterate Over Each Repository and Fetch Contributors
        contributor_counts = []
        for index, row in df.iterrows():
            owner, repo = row["owner"], row["repo"]
            contributor_counts.append(self.get_contributors(platform, owner, repo))

        # Add Results to DataFrame
        df[Metrics.CONTRIBUTOR.value] = contributor_counts