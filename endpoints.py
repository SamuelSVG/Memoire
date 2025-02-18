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

    GITLAB_SEARCH = "https://gitlab.com/api/v4/projects"
    @staticmethod
    def GITLAB_COMMITS_CONTRIBUTORS(owner, repo, branch):
        return f"https://gitlab.com/{owner}/{repo}/-/graphs/{branch}?format=json&ref_type=heads"
    @staticmethod
    def GITLAB_BRANCHES(id):
        return f"https://gitlab.com/api/v4/projects/{id}/repository/branches"
    @staticmethod
    def GITLAB_GRAPHQL():
        return "https://gitlab.com/api/graphql"
    @staticmethod
    def GITLAB_PULL_REQUESTS(owner, repo):
        return "{project(fullPath: " + f"\"{owner}/{repo}\"" + "){mergeRequests{count}}}"
    @staticmethod
    def GITLAB_LANGUAGE(id):
        return f"https://gitlab.com/api/v4/projects/{id}/languages"
    @staticmethod
    def GITLAB_LICENSE(id):
        return f"https://gitlab.com/api/v4/projects/{id}?license=true"
