from enum import Enum

class Metrics(Enum):
    COMMIT = "#commits"
    BRANCH = "#branches"
    CONTRIBUTOR = "#contributors"
    ISSUE = "#issues"
    PULL_REQUEST = "#pull_requests"
    LANGUAGE= "language"
    LICENSE= "license"
    SIZE = "size"