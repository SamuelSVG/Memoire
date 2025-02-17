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


    # Function to fetch repositories
    def fetch_repositories(self, page_num, platform=""):
        repositories = []
        for page in range(1, page_num):  # 10 pages × 100 repos per page = 1,000 repos
            print(f"Fetching page {page}...")
            params = {
                "q": "stars:>-1",
                "sort": "updated",  # Sort by most recently updated
                "order": "desc",  # Descending order
                "per_page": 100,  # Maximum allowed per page
                "page": page,  # Current page
            }
            response = requests.get(Endpoints.GITHUB_SEARCH.value, headers=self.headers, params=params)

            # Check for errors
            if response.status_code != 200:
                print(f"Error: {response.status_code}, {response.json()}")
                break

            data = response.json()
            repositories.extend(data['items'])
            time.sleep(5)

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