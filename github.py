from base_platform import BasePlatform
import requests
from endpoints import Endpoints
from request_types import RequestTypes
from metrics import Metrics

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
                    "default_branch": repo["default_branch"],
                    "#stars": repo["stargazers_count"],
                    "#forks": repo["forks_count"]
                }
                for repo in repositories
            ]
            return data
        return None

    def get_size(self, owner, repo, branch):
        """
        Function to fetch the size of a GitHub repository.
        :param owner: Owner of the repository.
        :param repo: Repository name.
        :param branch: Branch name.
        :return: Size of the repository.
        """
        url = Endpoints.GITHUB_SIZE(owner, repo, branch)
        try:
            response = self.request_with_retry(url, RequestTypes.GET, headers=self.headers)
            if not str(response.status_code).startswith('4') or str(response.status_code).startswith('5'):
                data = response.json()
                # Calculate total size of all blobs in the repository
                if "tree" in data:
                    total_size = sum(item.get("size", 0) for item in data["tree"] if item["type"] == "blob")
                    return total_size
                else:
                    return None
            else:
                self.logger.error(f"Error fetching {owner}/{repo}: {response.status_code}")
                return None
        except Exception as e:
            self.logger.exception(f"Exception fetching {owner}/{repo}: {e}")
            return None

    def add_size(self, df):
        """
        Function to add the size of a repository to a DataFrame of repositories.
        :param df: DataFrame of repositories.
        :return: DataFrame with the added size column.
        """
        size_counts = []
        for index, row in df.iterrows():
            owner, repo, branch = row["owner"], row["repo"], row["default_branch"]
            self.logger.info(f"Fetching data for {owner}/{repo}...")
            size_counts.append(self.get_size(owner, repo, branch))

        df[Metrics.SIZE.value] = size_counts