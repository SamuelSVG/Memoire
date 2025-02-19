# Headers for different platforms
from base_platform import BasePlatform
import time
import requests
import pandas as pd
from endpoints import Endpoints
from metrics import Metrics

class Gitlab(BasePlatform):
    def __init__(self, headers):
        self.headers = headers

    def fetch_page(self, page):
        params = {
            "visibility": "public",
            "order_by": "last_activity_at",  # Sort by most recently updated
            "sort": "desc",  # Descending order
            "per_page": 100,  # Maximum allowed per page
            "page": page,  # Current page
        }
        response = self.request_with_retry(Endpoints.GITLAB_SEARCH.value, headers=self.headers, params=params)
        return response.json()

    def fetch_repositories(self, page_num, platform=""):
        repositories = []
        for page in range(1, page_num):  # 10 pages × 100 repos per page = 1,000 repos
            self.logger.info(f"Fetching page {page}...")
            try:
                data = self.fetch_page(page)
                repositories.extend(data)
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Error fetching page {page}: {e}")
                break

        if repositories:
            data = [
                {
                    "platform": "GitLab",
                    "owner": repo["path_with_namespace"].rsplit("/", 1)[0],
                    "repo": repo["path"],
                    "id": repo["id"],
                    "created": repo["created_at"],
                    "updated": repo["last_activity_at"],
                    "default_branch": repo.get("default_branch", None),
                    "language": None,
                    "license": None,
                    "size": None,
                    "#stars": repo.get("star_count", None),
                    "#forks": repo.get("forks_count", None),
                }
                for repo in repositories
            ]
            return data
        return None

    def add_metric(self, df, metric, platform=None):
        # Iterate Over Each Repository and Fetch Metrics
        metric_counts = []
        for index, row in df.iterrows():
            match metric:
                case Metrics.COMMIT:
                    owner, repo, default_branch = row["owner"], row["repo"], row["default_branch"]
                    metric_counts.append(self.get_commits_contributors(owner,repo,default_branch))
                case Metrics.CONTRIBUTOR:
                    owner, repo, default_branch = row["owner"], row["repo"], row["default_branch"]
                    metric_counts.append(self.get_commits_contributors(owner,repo,default_branch))
                case Metrics.BRANCH:
                    id = row["id"]
                    metric_counts.append(self.get_branches(id))
                case Metrics.ISSUE:
                    owner, repo = row["owner"], row["repo"]
                    metric_counts.append(self.get_issues(owner,repo))
                case Metrics.PULL_REQUEST:
                    owner, repo = row["owner"], row["repo"]
                    metric_counts.append(self.get_pull_requests(owner,repo))
                case Metrics.LANGUAGE:
                    id = row["id"]
                    metric_counts.append(self.get_language(id))
                case Metrics.LICENSE:
                    id = row["id"]
                    metric_counts.append(self.get_license(id))

        # Add Results to DataFrame
        if metric == Metrics.COMMIT or metric == Metrics.CONTRIBUTOR:
            df[Metrics.COMMIT.value] = [count[0] if count else None for count in metric_counts]
            df[Metrics.CONTRIBUTOR.value] = [count[1] if count else None for count in metric_counts]
        else:
            df[metric.value] = metric_counts

    def get_commits_contributors(self, owner, repo, default_branch):
        url = Endpoints.GITLAB_COMMIT(owner, repo, default_branch)
        try:
            response = self.request_with_retry(url, headers=self.headers)
            if response.status_code == 200:
                data = response.json()
                unique_contributors = {commit["author_name"] for commit in data}
                return len(response.json()), len(unique_contributors)
            else:
                self.logger.error(f"Error fetching {owner}/{repo}: {response.status_code}")
                return None
        except Exception as e:
            self.logger.exception(f"Exception fetching {owner}/{repo}: {e}")
            return None

    def get_branches(self, id):
        url = Endpoints.GITLAB_BRANCH(id)
        total_branches = 0
        page = 1

        try:
            while True:
                response = self.request_with_retry(url, headers=self.headers, params={"page": page, "per_page": 100})
                if response.status_code == 200:
                    branches = response.json()
                    total_branches += len(branches)
                    if len(branches) < 100:
                        break
                    page += 1
                else:
                    self.logger.error(f"Error fetching {id}: {response.status_code}")
                    return None
            return total_branches
        except Exception as e:
            self.logger.exception(f"Exception fetching {id}: {e}")
            return None

    def get_issues(self, owner, repo):
        url = Endpoints.GITLAB_GRAPHQL()
        query = {"query": "{ project(fullPath: \"" + f"{owner}/{repo}" + "\") { issues { count } } }"}

        try:
            response = requests.post(url, json=query, headers=self.headers)
            if response.status_code == 200:
                return response.json()["data"]["project"]["issues"]["count"]
            else:
                self.logger.error(f"Error fetching {owner}/{repo}: {response.status_code}")
                return None
        except Exception as e:
            self.logger.exception(f"Exception fetching {owner}/{repo}: {e}")
            return None

    def get_pull_requests(self, owner, repo):
        url = Endpoints.GITLAB_GRAPHQL()
        query = {"query": "{ project(fullPath: \"" + f"{owner}/{repo}" + "\") { mergeRequests { count } } }"}

        try:
            response = requests.post(url, json=query, headers=self.headers)
            if response.status_code == 200:
                return response.json()["data"]["project"]["mergeRequests"]["count"]
            else:
                self.logger.error(f"Error fetching {owner}/{repo}: {response.status_code}")
                return None
        except Exception as e:
            self.logger.exception(f"Exception fetching {owner}/{repo}: {e}")
            return None

    def get_language(self, id):
        url = Endpoints.GITLAB_LANGUAGE(id)
        try:
            response = self.request_with_retry(url, headers=self.headers)
            if response.status_code == 200:
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
        url = Endpoints.GITLAB_LICENSE(id)
        try:
            response = self.request_with_retry(url, headers=self.headers)
            if response.status_code == 200:
                return response.json()["license"]["key"] if response.json()["license"] else None
            else:
                self.logger.error(f"Error fetching {id}: {response.status_code}")
                return None
        except Exception as e:
            self.logger.exception(f"Exception fetching {id}: {e}")
            return None
