from enum import Enum


class Endpoints(Enum):
    GITHUB_SEARCH = "https://api.github.com/search/repositories"
    GITHUB_GRAPHQL = "https://api.github.com/graphql"
    @staticmethod
    def GITHUB_REPO(owner, repo):
        return f"https://api.github.com/repos/{owner}/{repo}"
    @staticmethod
    def GITHUB_CLONE(owner, repo):
        return f"https://github.com/{owner}/{repo}.git"
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
    def GITEA_REPO(owner, repo):
        return f"https://gitea.com/api/v1/repos/{owner}/{repo}"
    @staticmethod
    def GITEA_CLONE(owner, repo):
        return f"https://gitea.com/{owner}/{repo}.git"
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
    def FORGEJO_REPO(owner, repo):
        return f"https://codeberg.org/api/v1/repos/{owner}/{repo}"
    @staticmethod
    def FORGEJO_CLONE(owner, repo):
        return f"https://codeberg.org/{owner}/{repo}.git"
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
    GITLAB_GRAPHQL = "https://gitlab.com/api/graphql"
    @staticmethod
    def GITLAB_REPO(id):
        return f"https://gitlab.com/api/v4/projects/{id}"
    @staticmethod
    def GITLAB_CLONE(owner, repo):
        return f"https://gitlab.com/{owner}/{repo}.git"
    @staticmethod
    def GITLAB_CONTRIBUTOR(owner, repo, branch):
        return f"https://gitlab.com/{owner}/{repo}/-/graphs/{branch}?format=json&ref_type=heads"
    @staticmethod
    def GITLAB_COMMIT(owner, repo):
        return f"https://gitlab.com/{owner}/{repo}"
    @staticmethod
    def GITLAB_PULL_REQUEST(owner, repo):
        return "{project(fullPath: " + f"\"{owner}/{repo}\"" + "){mergeRequests{count}}}"
    @staticmethod
    def GITLAB_LANGUAGE(id):
        return f"https://gitlab.com/api/v4/projects/{id}/languages"
    @staticmethod
    def GITLAB_LICENSE(id):
        return f"https://gitlab.com/api/v4/projects/{id}?license=true"


    BITBUCKET_SEARCH = "https://api.bitbucket.org/2.0/repositories"
    @staticmethod
    def BITBUCKET_REPO(owner, repo):
        return f"https://api.bitbucket.org/2.0/repositories/{owner}/{repo}"
    @staticmethod
    def BITBUCKET_CLONE(owner, repo):
        return f"https://bitbucket.org/{owner}/{repo}.git"
    @staticmethod
    def BITBUCKET_STAR(owner, repo):
        return f"https://api.bitbucket.org/2.0/repositories/{owner}/{repo}/watchers"
    @staticmethod
    def BITBUCKET_FORK(owner, repo):
        return f"https://api.bitbucket.org/2.0/repositories/{owner}/{repo}/forks"
    @staticmethod
    def BITBUCKET_PULL_REQUEST(owner, repo):
        return f"https://api.bitbucket.org/2.0/repositories/{owner}/{repo}/pullrequests?state=ALL"
    @staticmethod
    def BITBUCKET_ISSUE(owner, repo):
        return f"https://api.bitbucket.org/2.0/repositories/{owner}/{repo}/issues"
