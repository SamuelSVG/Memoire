from abc import ABC, abstractmethod
import requests
from endpoints import Endpoints


class BasePlatform(ABC):
    """
    Abstract base class that defines the interface for platform-specific matchers.
    """

    @abstractmethod
    def fetch_repositories(self, page_num, platform):
        """Load the platform's data"""
        pass

    # Function to Fetch Metrics from GitHub API
    def get_metric(self,platform,metric,owner,repo):
        temp = platform.name + "_" + metric.name
        url = getattr(Endpoints, temp)(owner, repo)

        try:
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                link_header = response.headers.get("Link", "")
                if 'rel="last"' in link_header:
                    last_page = link_header.split("page=")[-1].split(">")[0]
                    return int(last_page)
                return 1  # If no pagination, assume 1 entry
            elif response.status_code == 404:
                print(f"Repository not found: {owner}/{repo}")
                return None
            else:
                print(f"Error fetching {owner}/{repo}: {response.status_code}")
                return None
        except Exception as e:
            print(f"Exception fetching {owner}/{repo}: {e}")
            return None

    def add_metric(self, df, platform, metric):
        # Iterate Over Each Repository and Fetch Metrics
        metric_counts = []
        for index, row in df.iterrows():
            owner, repo = row["owner"], row["repo"]
            # print(f"Fetching data for {owner}/{repo}...")
            metric_counts.append(self.get_metric(platform,metric,owner,repo))

        # Add Results to DataFrame
        df[metric.value] = metric_counts
