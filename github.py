# Headers for different platforms
from base_platform import BasePlatform
import time
import requests
import pandas as pd
from endpoints import Endpoints
from metrics import Metrics

class GitHub(BasePlatform):
    def __init__(self, headers):
        self.headers = headers

    def fetch_page(self, page):
        params = {
            "q": "stars:>-1",
            "sort": "updated",
            "order": "desc",
            "per_page": 100,
            "page": page,
        }
        response = self.request_with_retry(Endpoints.GITHUB_SEARCH.value, headers=self.headers, params=params)
        return response.json()

    def fetch_repositories(self, page_num, platform=""):
        repositories = []
        for page in range(1, page_num):  # 10 pages × 100 repos per page = 1,000 repos
            self.logger.info(f"Fetching page {page}...")
            try:
                data = self.fetch_page(page)
                repositories.extend(data['items'])
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Error fetching page {page}: {e}")
                break

        if repositories:
            data = [
                {
                    "platform": "GitHub",
                    "owner": repo["owner"]["login"],
                    "repo": repo["name"],
                    "id": repo["id"],
                    "created": repo["created_at"],
                    "updated": repo["pushed_at"],
                    "language": repo["language"],
                    "license": repo["license"]["key"] if repo["license"] else None,
                    "size": repo["size"],
                    "#stars": repo["stargazers_count"],
                    "#forks": repo["forks_count"]
                }
                for repo in repositories
            ]
            return data
        return None