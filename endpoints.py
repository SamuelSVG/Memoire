from enum import Enum


class Endpoints(Enum):
    GITHUB_SEARCH = "https://api.github.com/search/repositories"
    @staticmethod
    def GITHUB_COMMIT(owner, repo):
        return f"https://api.github.com/repos/{owner}/{repo}/commits?per_page=1"
    @staticmethod
    def GITHUB_BRANCH(owner, repo):
        return f"https://api.github.com/repos/{owner}/{repo}/branches?per_page=1"
    @staticmethod
    def GITHUB_CONTRIBUTOR(owner, repo):
        return f"https://api.github.com/repos/{owner}/{repo}/contributors?per_page=1"
    @staticmethod
    def GITHUB_ISSUE(owner, repo):
        return f"https://api.github.com/repos/{owner}/{repo}/issues?state=all&per_page=1"
    @staticmethod
    def GITHUB_PULL_REQUEST(owner, repo):
        return f"https://api.github.com/repos/{owner}/{repo}/pulls?state=all&per_page=1"
    @staticmethod
    def GITHUB_SIZE(owner, repo, branch):
        return f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"


    GITEA_SEARCH = "https://gitea.com/api/v1/repos/search"
    @staticmethod
    def GITEA_COMMIT(owner, repo):
        return f"https://gitea.com/api/v1/repos/{owner}/{repo}/commits?limit=1"
    @staticmethod
    def GITEA_BRANCH(owner, repo):
        return f"https://gitea.com/api/v1/repos/{owner}/{repo}/branches?limit=1"
    @staticmethod
    def GITEA_CONTRIBUTOR(owner, repo):
        return f"https://gitea.com/{owner}/{repo}/activity/contributors/data"
    @staticmethod
    def GITEA_ISSUE(owner, repo):
        return f"https://gitea.com/api/v1/repos/{owner}/{repo}/issues?state=all&limit=1"
    @staticmethod
    def GITEA_PULL_REQUEST(owner, repo):
        return f"https://gitea.com/api/v1/repos/{owner}/{repo}/pulls?limit=1"
    @staticmethod
    def GITEA_SIZE(owner, repo, branch, page):
        return f"https://gitea.com/api/v1/repos/{owner}/{repo}/git/trees/{branch}?recursive=1&page={page}"


    FORGEJO_SEARCH = "https://codeberg.org/api/v1/repos/search"
    @staticmethod
    def FORGEJO_COMMIT(owner, repo):
        return f"https://codeberg.org/api/v1/repos/{owner}/{repo}/commits?limit=1"
    @staticmethod
    def FORGEJO_BRANCH(owner, repo):
        return f"https://codeberg.org/api/v1/repos/{owner}/{repo}/branches?limit=1"
    @staticmethod
    def FORGEJO_CONTRIBUTOR(owner, repo):
        return f"https://codeberg.org/{owner}/{repo}/activity/contributors/data"
    @staticmethod
    def FORGEJO_ISSUE(owner, repo):
        return f"https://codeberg.org/api/v1/repos/{owner}/{repo}/issues?state=all&limit=1"
    @staticmethod
    def FORGEJO_PULL_REQUEST(owner, repo):
        return f"https://codeberg.org/api/v1/repos/{owner}/{repo}/pulls?limit=1"
    @staticmethod
    def FORGEJO_SIZE(owner, repo, branch, page):
        return f"https://codeberg.org/api/v1/repos/{owner}/{repo}/git/trees/{branch}?recursive=1&page={page}"


    GITLAB_SEARCH = "https://gitlab.com/api/v4/projects"
    @staticmethod
    def GITLAB_CONTRIBUTOR(owner, repo, branch):
        return f"https://gitlab.com/{owner}/{repo}/-/graphs/{branch}?format=json&ref_type=heads"
    @staticmethod
    def GITLAB_COMMIT(owner, repo):
        return f"https://gitlab.com/{owner}/{repo}"
    @staticmethod
    def GITLAB_GRAPHQL():
        return "https://gitlab.com/api/graphql"
    @staticmethod
    def GITLAB_PULL_REQUEST(owner, repo):
        return "{project(fullPath: " + f"\"{owner}/{repo}\"" + "){mergeRequests{count}}}"
    @staticmethod
    def GITLAB_LANGUAGE(id):
        return f"https://gitlab.com/api/v4/projects/{id}/languages"
    @staticmethod
    def GITLAB_LICENSE(id):
        return f"https://gitlab.com/api/v4/projects/{id}?license=true"
