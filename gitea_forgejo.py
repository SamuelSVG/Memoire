from base_platform import BasePlatform
import requests
from endpoints import Endpoints
from request_types import RequestTypes
from metrics import Metrics
import math

class GiteaForgejo(BasePlatform):
    """
    This class defines the Gitea and Forgejo platform-specific matchers.
    """
    def __init__(self, headers):
        self.headers = headers


    def fetch_page(self, page, platform):
        """
        Function to fetch a page of repositories from the Gitea or Forgejo API.
        :param page: The page number to fetch.
        :param platform: The platform to fetch repositories from.
        :return: JSON response from the API.
        """
        params = {
            "q": "",  # Empty query to fetch all repositories
            "mode": "source",  # Fetch only source repositories
            "sort": "updated",  # Sort by most recently updated
            "order": "desc",
            "limit": 50,  # Maximum allowed per page
            "page": page
        }

        # Set the platform endpoint based on the platform name
        platform_endpoint = platform.name + "_SEARCH"
        url = getattr(Endpoints, platform_endpoint).value
        response = self.request_with_retry(url, RequestTypes.GET, headers=self.headers, params=params)
        return response.json()


    def fetch_repositories(self, page_num, platform=""):
        """
        Function to fetch a given number of pages of repositories from the Gitea or Forgejo API.
        :param page_num: Number of pages to fetch.
        :param platform: Platform to fetch repositories from.
        :return: List of dictionaries containing repository data.
        """
        repositories = []
        for page in range(1, page_num):  # 100 repos per page
            self.logger.info(f"Fetching page {page}...")
            try:
                data = self.fetch_page(page, platform)
                repositories.extend(data.get("data", []))
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Error fetching page {page}: {e}")
                break

        if repositories:
            # Extract relevant repository data
            data = [
                {
                    "platform": platform.value.capitalize(),
                    "owner": repo["owner"]["username"],
                    "repo": repo["name"],
                    "id": repo["id"],
                    "created": repo["created_at"],
                    "updated": repo["updated_at"],
                    "default_branch": repo["default_branch"],
                    "#stars": repo["stars_count"],
                    "#forks": repo["forks_count"]
                }
                for repo in repositories
            ]
            return data
        return None


    def get_contributors(self, platform, owner, repo):
        """
        Function to fetch the number of contributors for a given repository from the Gitea or Forgejo API.
        :param platform: The platform to fetch the contributors from.
        :param owner: Owner of the repository.
        :param repo: Repository name.
        :return: Number of contributors for the given repository.
        """
        url = getattr(Endpoints, platform.name + "_CONTRIBUTOR")(owner, repo)
        try:
            response = self.request_with_retry(url, RequestTypes.GET, headers=self.headers)

            if not str(response.status_code).startswith('4') or str(response.status_code).startswith('5'):
                return len(response.json())-1
            else:
                self.logger.error(f"Error fetching {owner}/{repo}: {response.status_code}")
                return None
        except Exception as e:
            self.logger.exception(f"Exception fetching {owner}/{repo}: {e}")
            return None


    def add_contributors(self, df, platform):
        """
        Function to add the number of contributors to a DataFrame of repositories.
        :param df: DataFrame of repositories.
        :param platform: Platform to fetch the contributors from.
        :return: DataFrame with the added contributor column.
        """
        contributor_counts = []
        for index, row in df.iterrows():
            owner, repo = row["owner"], row["repo"]
            contributor_counts.append(self.get_contributors(platform, owner, repo))

        df[Metrics.CONTRIBUTOR.value] = contributor_counts

    def get_size(self, platform, owner, repo, branch):
        """
        Function to fetch the size of a repository from the Gitea or Forgejo API.
        :param platform: The platform to fetch the size from.
        :param owner: Owner of the repository.
        :param repo: Repository name.
        :param branch: Branch name.
        :return: Size of the repository.
        """
        total_size = 0
        page = 1
        try:
            while True:
                url = getattr(Endpoints, platform.name + "_SIZE")(owner, repo, branch, page)
                response = self.request_with_retry(url,RequestTypes.GET,headers=self.headers)

                if not str(response.status_code).startswith('4') or str(response.status_code).startswith('5'):
                    data = response.json()
                    # Calculate total size of all blobs in the repository
                    if "tree" in data:
                        total_size += sum(item.get("size", 0) for item in data["tree"] if item["type"] == "blob")

                    # Check if there are more pages
                    if "x-total-count" in response.headers:
                        total_objects = int(response.headers["x-total-count"])
                        total_pages = math.ceil(total_objects / 1000)
                        if page >= total_pages:
                            break  # Stop if all pages are fetched
                    else:
                        break  # Stop if there's no pagination info
                    page += 1  # Move to the next page
                else:
                    self.logger.error(f"Error fetching {owner}/{repo}: {response.status_code}")
                    return None
            return total_size

        except Exception as e:
            self.logger.exception(f"Exception fetching {owner}/{repo}: {e}")
            return None

    def add_size(self, df, platform):
        """
        Function to add the size of a repository to a DataFrame of repositories.
        :param df: DataFrame of repositories.
        :param platform: Platform to fetch the size from.
        :return: DataFrame with the added size column.
        """
        size_counts = []
        for index, row in df.iterrows():
            owner, repo, branch = row["owner"], row["repo"], row["default_branch"]
            self.logger.info(f"Fetching data for {owner}/{repo}...")
            size_counts.append(self.get_size(platform, owner, repo, branch))

        df[Metrics.SIZE.value] = size_counts