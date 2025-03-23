from enum import Enum

class Metrics(Enum):
    COMMIT = "#commits"
    BRANCH = "#branches"
    STAR = "#stars"
    FORK = "#forks"
    CONTRIBUTOR = "#contributors"
    ISSUE = "#issues"
    PULL_REQUEST = "#pull_requests"
    LANGUAGE= "language"
    LICENSE= "license"
    SIZE = "size"
    UPDATED= "updated"