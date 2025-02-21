from base_platform import BasePlatform
import requests
from endpoints import Endpoints
from request_types import RequestTypes

class GitHub(BasePlatform):
    """
    This class defines the GitHub platform-specific matcher.
    """
    def __init__(self, headers):
        self.headers = headers


    def fetch_page(self, page):
        """
        Function to fetch a page of repositories from the GitHub API.
        :param page: The page number to fetch.
        :return: JSON response from the API.
        """
        params = {
            "q": "stars:>-1", # Empty query to fetch all repositories
            "sort": "updated",
            "order": "desc",
            "per_page": 100, # Maximum allowed per page
            "page": page, # Page number
        }
        response = self.request_with_retry(Endpoints.GITHUB_SEARCH.value, RequestTypes.GET, headers=self.headers, params=params)
        return response.json()


    def fetch_repositories(self, page_num, platform=""):
        """
        Function to fetch a given number of pages of repositories from the GitHub API.
        :param page_num: Number of pages to fetch.
        :param platform: Platform to fetch repositories from.
        :return: List of dictionaries containing repository data.
        """
        repositories = []
        for page in range(1, page_num):  # 100 repos per page
            self.logger.info(f"Fetching page {page}...")
            try:
                data = self.fetch_page(page)
                repositories.extend(data['items'])
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Error fetching page {page}: {e}")
                break

        # Extract relevant repository data
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