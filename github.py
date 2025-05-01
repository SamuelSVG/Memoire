from base_platform import BasePlatform
import requests
from endpoints import Endpoints
from request_types import RequestTypes
from metrics import Metrics
from datetime import datetime, timedelta
from excluded_names import EXCLUDED_NAMES
import time

class GitHub(BasePlatform):
    """
    This class defines the GitHub platform-specific matcher.
    """
    def __init__(self, headers):
        self.headers = headers


    def fetch_page(self, page, creation_date):
        """
        Function to fetch a page of repositories from the GitHub API.
        :param page: The page number to fetch.
        :return: JSON response from the API.
        """
        today = datetime.now().strftime('%Y-%m-%d')
        params = {
            "q": f"created:<={creation_date} pushed:{today} fork:false",
            "per_page": 100,
            "page": page,
            "sort": "updated",  # Optional: sort by recent updates
            "order": "desc"
        }
        response = self.request_with_retry(Endpoints.GITHUB_SEARCH.value, RequestTypes.GET, headers=self.headers, params=params)
        return response.json()


    def fetch_repositories(self, target, creation_date=None, platform="GitHub"):
        """
        Function to fetch a given number of pages of repositories from the GitHub API.
        :param target: Number of pages to fetch.
        :param platform: Platform to fetch repositories from.
        :return: List of dictionaries containing repository data.
        """
        if target > 1000:
            raise ValueError("n_repositories must be less than or equal to 1000 because GitHub has a hard limit on"
                             "the search endpoint.")

        if creation_date is None:
            creation_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

        if platform is None:
            raise ValueError("Platform must be specified.")

        repositories = []
        page = 1

        while len(repositories) < target:
            if page > 10:
                time.sleep(30)
                page = 1
            self.logger.info(f"Fetching page {page}...")
            try:
                data = self.fetch_page(page, creation_date)
                repos_in_page = data.get("items", [])
                if not repos_in_page:
                    self.logger.info(f"No more repositories found on page {page}.")
                    break

                for repo in repos_in_page:
                    if ((repo["owner"]["login"], repo["name"]) not in [(r["owner"]["login"], r["name"]) for r in repositories]
                            and not any(bad_word in repo["owner"]["login"].lower() for bad_word in EXCLUDED_NAMES)
                            and not any(bad_word in repo["name"].lower() for bad_word in EXCLUDED_NAMES)):

                        repositories.append(repo)
                        if len(repositories) >= target:
                            break

                page += 1

            except requests.exceptions.RequestException as e:
                self.logger.error(f"Error fetching page {page}: {e}")
                break

        # Extract relevant repository data
        if repositories:
            data = [
                {
                    "platform": platform,
                    "owner": repo["owner"]["login"],
                    "repo": repo["name"],
                    "id": repo["id"],
                    "created": repo["created_at"],
                    "updated": repo["pushed_at"],
                    "default_branch": repo["default_branch"],
                    "#stars": repo["stargazers_count"],
                    "#forks": repo["forks_count"]
                }
                for repo in repositories[:target]
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

    def get_issues(self, owner, repo):
        """
        Function to fetch the number of issues for a given repository from the GitLab API.
        :param owner: Owner of the repository.
        :param repo: Repository name.
        :return: Number of issues for the given repository.
        """
        url = Endpoints.GITHUB_GRAPHQL.value
        query = {
            "query": "{ repository(owner: \"" + f"{owner}" + "\", name: \"" + f"{repo}" + "\") { issues { totalCount } } }"
        }

        try:
            response = self.request_with_retry(url, RequestTypes.POST, params=query, headers=self.headers)
            if not str(response.status_code).startswith('4'):
                return response.json()["data"]["repository"]["issues"]["totalCount"]
            else:
                self.logger.error(f"Error fetching {owner}/{repo}: {response.status_code}")
                return None
        except Exception as e:
            self.logger.exception(f"Exception fetching {owner}/{repo}: {e}")
            return None


    def get_pull_requests(self, owner, repo):
        """
        Function to fetch the number of pull requests for a given repository from the GitLab API.
        :param owner: Owner of the repository.
        :param repo: Repository name.
        :return: Number of pull requests for the given repository.
        """
        url = Endpoints.GITHUB_GRAPHQL.value
        query = {
            "query": "{ repository(owner: \"" + f"{owner}" + "\", name: \"" + f"{repo}" + "\") { pullRequests { totalCount } } }"
        }

        try:
            response = self.request_with_retry(url, RequestTypes.POST, params=query, headers=self.headers)
            if not str(response.status_code).startswith('4'):
                return response.json()["data"]["repository"]["pullRequests"]["totalCount"]
            else:
                self.logger.error(f"Error fetching {owner}/{repo}: {response.status_code}")
                return None
        except Exception as e:
            self.logger.exception(f"Exception fetching {owner}/{repo}: {e}")
            return None

    def add_metric(self, df, metric):
        """
        Function to add a given metric to a DataFrame of repositories.
        :param df: DataFrame of repositories.
        :param metric: Metric to fetch.
        :param platform: Platform to fetch the metric from.
        :return: DataFrame with the added metric column.
        """
        metric_counts = []
        for index, row in df.iterrows():
            self.logger.info(f"Fetching data for {row["owner"]}/{row["repo"]}...")

            # Gitlab requires to use multiple different endpoints to fetch different metrics so we need to use a switch case
            match metric:
                case Metrics.SIZE:
                    owner, repo, branch = row["owner"], row["repo"], row["default_branch"]
                    metric_counts.append(self.get_size(owner,repo,branch))
                case Metrics.ISSUE:
                    owner, repo = row["owner"], row["repo"]
                    metric_counts.append(self.get_issues(owner, repo))
                case Metrics.PULL_REQUEST:
                    owner, repo = row["owner"], row["repo"]
                    metric_counts.append(self.get_pull_requests(owner, repo))

        df[metric.value] = metric_counts