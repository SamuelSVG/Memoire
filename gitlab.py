from base_platform import BasePlatform
import requests
from endpoints import Endpoints
from metrics import Metrics
from request_types import RequestTypes
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec
from datetime import datetime, timedelta
from excluded_names import EXCLUDED_NAMES

class Gitlab(BasePlatform):
    """
    This class defines the GitLab platform-specific matcher.
    """
    def __init__(self, headers):
        self.headers = headers

    # Set up the Chrome driver
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    service = Service(ChromeDriverManager().install())

    def fetch_page(self, page):
        """
        Function to fetch a page of repositories from the GitLab API.
        :param page: The page number to fetch.
        :return: JSON response from the API.
        """
        params = {
            "visibility": "public",
            "order_by": "last_activity_at",  # Sort by most recently updated
            "sort": "desc",
            "per_page": 100,
            "page": page,
        }
        response = self.request_with_retry(Endpoints.GITLAB_SEARCH.value, RequestTypes.GET, headers=self.headers, params=params)
        return response.json()

    def fetch_repositories(self, target, creation_date=15, platform="GitLab"):
        """
        Fetch a target number of acceptable repositories (older than creation_date) from GitLab.

        :param target: Number of repositories you want to fetch.
        :param creation_date: Date cutoff for repo creation (default: 15 days ago).
        :param platform: Platform name (default: GitLab).
        :return: List of dictionaries containing repository data.
        """
        repositories = []
        page = 1

        while len(repositories) < target:
            self.logger.info(f"Fetching page {page}...")
            try:
                data = self.fetch_page(page)
                page += 1

                if not data:
                    self.logger.info("No more repositories found.")
                    break

                # Filter out forks and repositories created too recently
                for repo in data:
                    if "forked_from_project" not in repo:
                        created_at = datetime.fromisoformat(repo["created_at"].replace("Z", "+00:00")).date()
                        creation_cutoff = datetime.fromisoformat(repo["last_activity_at"].replace("Z", "+00:00")).date() - timedelta(days=creation_date)
                        if ((repo["path_with_namespace"], repo["path"]) not in [(r["path_with_namespace"], r["path"]) for r in repositories]
                             and (created_at <= creation_cutoff
                             and not any(bad_word in repo["path_with_namespace"].lower() for bad_word in EXCLUDED_NAMES))
                             and not any(bad_word in repo["path"].lower() for bad_word in EXCLUDED_NAMES)):
                            repositories.append(repo)
                            if len(repositories) >= target:
                                break

            except requests.exceptions.RequestException as e:
                self.logger.error(f"Error fetching page {page}: {e}")
                break

        # Extract relevant repository data
        if repositories:
            data = [
                {
                    "platform": platform,
                    "owner": repo["path_with_namespace"].rsplit("/", 1)[0],
                    "repo": repo["path"],
                    "id": repo["id"],
                    "created": repo["created_at"],
                    "updated": repo["last_activity_at"],
                    "default_branch": repo.get("default_branch", None),
                    "#stars": repo.get("star_count", None),
                    "#forks": repo.get("forks_count", None),
                }
                for repo in repositories[:target]  # Limit strictly to the target number
            ]
            return data
        return None

    def add_metric(self, df, metric, platform=None):
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

            match metric:
                case Metrics.COMMIT:
                    owner, repo = row["owner"], row["repo"]
                    metric_counts.append(self.get_commits_branches(owner,repo))
                case Metrics.BRANCH:
                    owner, repo = row["owner"], row["repo"]
                    metric_counts.append(self.get_commits_branches(owner, repo))
                case Metrics.CONTRIBUTOR:
                    owner, repo, default_branch = row["owner"], row["repo"], row["default_branch"]
                    metric_counts.append(self.get_contributors(owner,repo, default_branch))
                case Metrics.ISSUE:
                    owner, repo = row["owner"], row["repo"]
                    metric_counts.append(self.get_issues_pull_requests(owner,repo))
                case Metrics.PULL_REQUEST:
                    owner, repo = row["owner"], row["repo"]
                    metric_counts.append(self.get_issues_pull_requests(owner,repo))
                case Metrics.LANGUAGE:
                    id = row["id"]
                    metric_counts.append(self.get_language(id))
                case Metrics.LICENSE:
                    id = row["id"]
                    metric_counts.append(self.get_license(id))

        # Add Results to DataFrame
        if metric == Metrics.COMMIT or metric == Metrics.BRANCH:
            df[Metrics.COMMIT.value] = [count[0] if count else None for count in metric_counts]
            df[Metrics.BRANCH.value] = [count[1] if count else None for count in metric_counts]
        elif metric == Metrics.ISSUE or metric == Metrics.PULL_REQUEST:
            df[Metrics.ISSUE.value] = [count[0] if count else None for count in metric_counts]
            df[Metrics.PULL_REQUEST.value] = [count[1] if count else None for count in metric_counts]
        else:
            df[metric.value] = metric_counts


    def get_commits_branches(self, owner, repo):
        """
        Function to fetch the number of commits and branches for a given repository from the GitLab API.
        :param owner: Owner of the repository.
        :param repo: Repository name.
        :return: Number of commits and branches for the given repository.
        """
        url = Endpoints.GITLAB_COMMIT(owner, repo)

        # We get the number of commits and branches by scraping the GitLab page as there is no api endpoint
        driver = webdriver.Chrome(service=self.service, options=self.options)
        driver.get(url)

        # Wait for the elements to load
        try:
            WebDriverWait(driver, 15).until(
                ec.presence_of_all_elements_located((By.CLASS_NAME, "project-stat-value"))
            )
        except Exception as e:
            self.logger.error(f"Error: {e}")
            driver.quit()
            return []

        # Find all elements with class "project-stat-value"
        elements = driver.find_elements(By.CLASS_NAME, "project-stat-value")
        # Get the commit count and branch count
        values = [el.text.strip() for el in elements[:2]]
        driver.quit()

        return values


    def get_contributors(self, owner, repo, default_branch):
        """
        Function to fetch the number of contributors for a given repository from the GitLab API.
        :param owner: Owner of the repository.
        :param repo: Repository name.
        :param default_branch: Default branch of the repository.
        :return: Number of contributors for the given repository.
        """
        url = Endpoints.GITLAB_CONTRIBUTOR(owner, repo, default_branch)
        try:
            response = self.request_with_retry(url, RequestTypes.GET, headers=self.headers)
            if not str(response.status_code).startswith('4'):
                data = response.json()
                unique_contributors = {commit["author_name"] for commit in data}
                return len(unique_contributors)
            else:
                self.logger.error(f"Error fetching {owner}/{repo}: {response.status_code}")
                return None
        except Exception as e:
            self.logger.exception(f"Exception fetching {owner}/{repo}: {e}")
            return None


    def get_issues(self, owner, repo):
        """
        Function to fetch the number of issues for a given repository from the GitLab API.
        :param owner: Owner of the repository.
        :param repo: Repository name.
        :return: Number of issues for the given repository.
        """
        url = Endpoints.GITLAB_GRAPHQL.value
        query = {"query": "{ project(fullPath: \"" + f"{owner}/{repo}" + "\") { issues { count } } }"}

        try:
            response = self.request_with_retry(url, RequestTypes.POST, params=query, headers=self.headers)
            if not str(response.status_code).startswith('4'):
                return response.json()["data"]["project"]["issues"]["count"]
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
        url = Endpoints.GITLAB_GRAPHQL.value
        query = {"query": "{ project(fullPath: \"" + f"{owner}/{repo}" + "\") { mergeRequests { count } } }"}

        try:
            response = self.request_with_retry(url, RequestTypes.POST, params=query, headers=self.headers)
            if not str(response.status_code).startswith('4'):
                return response.json()["data"]["project"]["mergeRequests"]["count"]
            else:
                self.logger.error(f"Error fetching {owner}/{repo}: {response.status_code}")
                return None
        except Exception as e:
            self.logger.exception(f"Exception fetching {owner}/{repo}: {e}")
            return None

    def get_issues_pull_requests(self, owner, repo):
        """
        Function to fetch the number of issues and pull requests for a given repository from the GitLab API.
        :param owner: Owner of the repository.
        :param repo: Repository name.
        :return: Tuple containing the number of issues and pull requests for the given repository.
        """
        url = Endpoints.GITLAB_GRAPHQL.value
        query = {"query": "{ project(fullPath: \"" + f"{owner}/{repo}" + "\") { issues { count } mergeRequests { count } } }"}

        try:
            response = self.request_with_retry(url, RequestTypes.POST, params=query, headers=self.headers)
            if not str(response.status_code).startswith('4'):
                return (response.json()["data"]["project"]["issues"]["count"],
                        response.json()["data"]["project"]["mergeRequests"]["count"])
            else:
                self.logger.error(f"Error fetching {owner}/{repo}: {response.status_code}")
                return None
        except Exception as e:
            self.logger.exception(f"Exception fetching {owner}/{repo}: {e}")
            return None

    def get_language(self, id):
        """
        Function to fetch the main language for a given repository from the GitLab API.
        :param id: Repository ID.
        :return: Main language of the repository.
        """
        url = Endpoints.GITLAB_LANGUAGE(id)
        try:
            response = self.request_with_retry(url, RequestTypes.GET, headers=self.headers)
            if not str(response.status_code).startswith('4'):
                languages = response.json()
                if languages:
                    main_language = next(iter(languages))
                    return main_language
                return None
            else:
                self.logger.error(f"Error fetching {id}: {response.status_code}")
                return None
        except Exception as e:
            self.logger.exception(f"Exception fetching {id}: {e}")
            return None


    def get_license(self, id):
        """
        Function to fetch the license for a given repository from the GitLab API.
        :param id: Repository ID.
        :return: License of the repository.
        """
        url = Endpoints.GITLAB_LICENSE(id)
        try:
            response = self.request_with_retry(url, RequestTypes.GET, headers=self.headers)
            if not str(response.status_code).startswith('4'):
                return response.json()["license"]["key"] if response.json()["license"] else None
            else:
                self.logger.error(f"Error fetching {id}: {response.status_code}")
                return None
        except Exception as e:
            self.logger.exception(f"Exception fetching {id}: {e}")
            return None
