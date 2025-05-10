from base_platform import BasePlatform
import requests
from endpoints import Endpoints
from request_types import RequestTypes
from metrics import Metrics
import math
from datetime import datetime, timedelta
from excluded_names import EXCLUDED_NAMES

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

    def fetch_repositories(self, target, creation_date=None, platform=None):
        """
        Fetch a target number of acceptable repositories (older than creation_date) from Gitea/Forgejo.

        :param target: Number of repositories you want to fetch.
        :param creation_date: Date cutoff for repo creation (default: 30 days ago).
        :param platform: Platform to fetch from.
        :return: List of repositories' data.
        """
        if creation_date is None:
            creation_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

        if platform is None:
            raise ValueError("Platform must be specified.")

        #creation_cutoff = datetime.strptime(creation_date, '%Y-%m-%d')

        repositories = []
        page = 1

        while len(repositories) < target:
            self.logger.info(f"Fetching page {page}...")
            try:
                data = self.fetch_page(page, platform)
                repos_in_page = data.get("data", [])

                if not repos_in_page:
                    self.logger.info("No more repositories found.")
                    break

                for repo in repos_in_page:
                    created_at = datetime.fromisoformat(repo.get("created_at").replace("Z", "+00:00")).date()
                    creation_cutoff = datetime.fromisoformat(repo.get("updated_at").replace("Z", "+00:00")).date() - timedelta(days=15)
                    if ((repo["owner"]["username"], repo["name"]) not in [(r["owner"]["username"], r["name"]) for r in repositories]
                            and (created_at <= creation_cutoff
                            and not any(bad_word in repo["owner"]["username"].lower() for bad_word in EXCLUDED_NAMES))
                            and not any(bad_word in repo["name"].lower() for bad_word in EXCLUDED_NAMES)):
                        repositories.append(repo)
                        if len(repositories) >= target:
                            break

                page += 1

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
                    "default_branch": repo.get("default_branch"),
                    "#stars": repo.get("stars_count"),
                    "#forks": repo.get("forks_count"),
                }
                for repo in repositories[:target]  # ensure no extra repos
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