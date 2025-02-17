from enum import Enum


class Endpoints(Enum):
    GITHUB_SEARCH = "https://api.github.com/search/repositories"
    @staticmethod
    def GITHUB_COMMITS(owner, repo):
        return f"https://api.github.com/repos/{owner}/{repo}/commits?per_page=1"
    @staticmethod
    def GITHUB_BRANCHES(owner, repo):
        return f"https://api.github.com/repos/{owner}/{repo}/branches?per_page=1"
    @staticmethod
    def GITHUB_CONTRIBUTORS(owner, repo):
        return f"https://api.github.com/repos/{owner}/{repo}/contributors?per_page=1"
    @staticmethod
    def GITHUB_ISSUES(owner, repo):
        return f"https://api.github.com/repos/{owner}/{repo}/issues?per_page=1"
    @staticmethod
    def GITHUB_PULL_REQUESTS(owner, repo):
        return f"https://api.github.com/repos/{owner}/{repo}/pulls?per_page=1"

    GITEA_SEARCH = "https://gitea.com/api/v1/repos/search"
    @staticmethod
    def GITEA_COMMITS(owner, repo):
        return f"https://gitea.com/api/v1/repos/{owner}/{repo}/commits?limit=1"
    @staticmethod
    def GITEA_BRANCHES(owner, repo):
        return f"https://gitea.com/api/v1/repos/{owner}/{repo}/branches?limit=1"
    @staticmethod
    def GITEA_CONTRIBUTORS(owner, repo):
        return f"https://gitea.com/{owner}/{repo}/activity/contributors/data"
    @staticmethod
    def GITEA_ISSUES(owner, repo):
        return f"https://gitea.com/api/v1/repos/{owner}/{repo}/issues?limit=1"
    @staticmethod
    def GITEA_PULL_REQUESTS(owner, repo):
        return f"https://gitea.com/api/v1/repos/{owner}/{repo}/pulls?limit=1"

    FORGEJO_SEARCH = "https://codeberg.org/api/v1/repos/search"
    @staticmethod
    def FORGEJO_COMMITS(owner, repo):
        return f"https://codeberg.org/api/v1/repos/{owner}/{repo}/commits?limit=1"
    @staticmethod
    def FORGEJO_BRANCHES(owner, repo):
        return f"https://codeberg.org/api/v1/repos/{owner}/{repo}/branches?limit=1"
    @staticmethod
    def FORGEJO_CONTRIBUTORS(owner, repo):
        return f"https://codeberg.org/{owner}/{repo}/activity/contributors/data"
    @staticmethod
    def FORGEJO_ISSUES(owner, repo):
        return f"https://codeberg.org/api/v1/repos/{owner}/{repo}/issues?limit=1"
    @staticmethod
    def FORGEJO_PULL_REQUESTS(owner, repo):
        return f"https://codeberg.org/api/v1/repos/{owner}/{repo}/pulls?limit=1"
