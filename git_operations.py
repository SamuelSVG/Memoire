import shutil
import git
import os
import subprocess
import json
from endpoints import Endpoints


def clone_repository(owner, repo, platform):
    """
    Function to clone a repository from a platform-specific API.
    :param owner: Owner of the repository.
    :param repo: Repository name.
    :param platform: Platform to fetch the repository from.
    """
    temp = platform.name + "_CLONE"
    url = getattr(Endpoints, temp)(owner, repo)
    try:
        target_path = os.path.join(os.getcwd(), "temp")
        print(f"Cloning {url} into {target_path}...")
        git.Repo.clone_from(url, target_path)
        print("Clone successful!")

    except Exception as e:
        print(f"Error cloning repository: {e}")

def get_repo_size(repo_path):
    """
    Function to retrieve the size of a repository on disk.
    :param repo_path: Path to the repository.
    :return: Size of the repository.
    """
    total_size = 0
    for dirpath, _, filenames in os.walk(repo_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):  # Avoid broken symlinks
                total_size += os.path.getsize(fp)
    print(total_size)
    return total_size  # Returns size in bytes


def get_repo_license(repo_path):
    """
    Function to detect the license of a repository using the 'licensee' tool.
    :param repo_path: Path to the repository.
    :return: License of the repository.
    """
    try:
        # Try to locate 'licensee' in the system's PATH
        licensee_path = shutil.which("licensee")
        if not licensee_path:
            print("Not found")
            raise FileNotFoundError("The 'licensee' tool is not installed or not in the system's PATH.")

        result = subprocess.run(
            [licensee_path, "detect", repo_path, "--json"],
            capture_output=True,
            text=True,
            check=True
        )

        data = json.loads(result.stdout)
        license_name = data.get("licenses", [])[0].get("spdx_id", "No license detected")
        return license_name
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None
    except subprocess.CalledProcessError as e:
        return None

def get_language_distribution(repo_path):
    """
    Function to detect the language distribution of a repository using the 'github-linguist' tool.
    :param repo_path: Path to the repository.
    :return: Language distribution of the repository.
    """
    try:
        linguist_path = shutil.which("github-linguist")
        if not linguist_path:
            raise FileNotFoundError("The 'licensee' tool is not installed or not in the system's PATH.")

        result = subprocess.run(
            [linguist_path, repo_path, "--json"],
            capture_output=True,
            text=True,
            check=True
        )

        output = result.stdout.strip()
        if output == "{}":
            print("No language distribution detected.")
            return None, None
        else:
            data = json.loads(output)
            main_language = max(data, key=lambda k: float(data[k]["percentage"]))
            return main_language, json.dumps(data)

    except FileNotFoundError:
        print("Error: GitHub Linguist command not found. Ensure it's installed.")
        return "Linguist not found"
    except subprocess.CalledProcessError as e:
        print(f"No language distribution detected: {e}")
        return None, None

def get_commit_count(repo_path):
    """Get the number of commits in a Git repository."""
    try:
        # Run 'git rev-list --count HEAD' to count the commits
        result = subprocess.run(
            ["git", "rev-list", "--count", "HEAD"],
            cwd=repo_path,  # Run the command inside the repo
            capture_output=True,
            text=True,
            check=True
        )

        commit_count = result.stdout.strip()  # Remove extra whitespace
        return commit_count

    except subprocess.CalledProcessError as e:
        print(f"Error getting commit count: {e}")
        return None

def get_branch_count(repo_path):
    """Get the number of branches in a Git repository."""
    try:
        # Run 'git branch --list' to get all branches and count them
        result = subprocess.run(
            ["git", "branch", "-a"],
            cwd=repo_path,  # Run the command inside the repo
            capture_output=True,
            text=True,
            check=True
        )

        # Split the output by newlines and count non-empty branches
        branches = result.stdout.strip().split("\n")
        branch_count = len([branch for branch in branches if branch.strip()]) #-2 pour kick current et head
        if branch_count > 2: #Cas où on a 0 branche (dépot vide)
            branch_count -= 2

        return branch_count

    except subprocess.CalledProcessError as e:
        print(f"Error getting branch count: {e}")
        return "Error getting branch count"

def get_contributor_count(repo_path):
    """Get the number of contributors in a Git repository using git shortlog."""
    try:
        # Run 'git shortlog -s -n --all | sort | uniq | wc -l' to get the count of unique contributors
        result = subprocess.run(
            "git shortlog -s -n --all | sort | uniq | wc -l",
            cwd=repo_path,  # Run the command inside the repo
            capture_output=True,
            text=True,
            shell=True,  # Enable shell to use pipes and multiple commands
            check=True
        )

        # The result will be a single number output by wc -l
        contributor_count = int(result.stdout.strip())

        return contributor_count

    except subprocess.CalledProcessError as e:
        print(f"Error getting contributor count: {e}")
        return "Error getting contributor count"

def delete_directory(repo_path):
    try:
        if not os.path.exists(repo_path):
            print(f"Directory '{repo_path}' does not exist.")
            return False
        else:
            shutil.rmtree(repo_path)
            print(f"Directory '{repo_path}' deleted.")
    except Exception as e:
        print(f"Error deleting directory contents: {e}")
        return False

def add_git_metrics(df, platform):
    # Ensure the necessary columns exist in the DataFrame
    columns = ["size", "license", "main_language", "language_distribution", "#commits", "#branches", "#contributors"]
    for column in columns:
        if column not in df.columns:
            df[column] = None

    for index, row in df.iterrows():
        owner, repo = row["owner"], row["repo"]
        try:
            repo_path = os.path.join(os.getcwd(), "temp")
            clone_repository(owner, repo, platform)
            df.at[index, "size"] = get_repo_size(repo_path)
            df.at[index, "license"] = get_repo_license(repo_path)
            df.at[index, "main_language"], df.at[index, "language_distribution"] = get_language_distribution(repo_path)
            df.at[index, "#commits"] = get_commit_count(repo_path)
            df.at[index, "#branches"] = get_branch_count(repo_path)
            df.at[index, "#contributors"] = get_contributor_count(repo_path)
            delete_directory(repo_path)
        except Exception as e:
            print(f"Error processing repository '{owner}/{repo}': {e}")