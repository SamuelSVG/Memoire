import logging
import shutil
import git
import os
import subprocess
import json
from endpoints import Endpoints
import git_tools
import sys
import platform as plat
import stat

# Set up logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

def setup_tools_containers():
    """Set up Docker containers with GitHub-Linguist and Licensee installed."""
    git_tools.check_docker()
    git_tools.build_licensee_image()
    git_tools.build_linguist_image()

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
        if os.path.exists(target_path):
            shutil.rmtree(target_path)
        logging.info(f"Cloning {url} into {target_path}...")
        git.Repo.clone_from(url, target_path)
        logging.info("Clone successful!")
        return True

    except Exception as e:
        logging.error(f"Error cloning repository: {e}")
        return False

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
    return total_size  # Returns size in bytes


def get_repo_license(repo_path):
    """
    Function to detect the license of a repository using the 'licensee' tool.
    :param repo_path: Path to the repository.
    :return: License of the repository.
    """
    try:
        data = git_tools.analyze_directory("licensee",repo_path)
        license_name = data.get("licenses", [])[0].get("spdx_id", "No license detected")
        return license_name
    except subprocess.CalledProcessError as e:
        logging.error(f"Error detecting license: {e}")
        return None

def get_language_distribution(repo_path):
    """
    Function to detect the language distribution of a repository using the 'github-linguist' tool.
    :param repo_path: Path to the repository.
    :return: Language distribution of the repository.
    """
    try:
        data = git_tools.analyze_directory("linguist",repo_path)
        if data == {}:
            logging.info("No language distribution detected.")
            return None, None
        else:
            main_language = max(data, key=lambda k: float(data[k]["percentage"]))
            return main_language, json.dumps(data)

    except subprocess.CalledProcessError as e:
        logging.error(f"Error when analyzing language distribution: {e}")
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
        logging.error(f"Error getting commit count: {e}")
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
        logging.error(f"Error getting branch count: {e}")
        return None

def get_contributor_count(repo_path):
    """Get the number of contributors in a Git repository using git shortlog."""
    try:
        if plat.system() == "Windows":
            # PowerShell command for Windows
            git_command = "(git shortlog -s -n --all | Sort-Object | Get-Unique | Measure-Object -Line).Lines"
            result = subprocess.run(
                ["powershell", "-Command", git_command],  # Run the command in PowerShell
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True
            )
        else:
            # Command for macOS/Linux
            git_command = "git shortlog -s -n --all | sort | uniq | wc -l"
            result = subprocess.run(
                git_command,
                cwd=repo_path,
                capture_output=True,
                text=True,
                shell=True,
                check=True
            )

        # Extract just the number
        contributor_count = int(result.stdout.strip())
        return contributor_count

    except subprocess.CalledProcessError as e:
        logging.error(f"Error getting contributor count: {e}")
        return None

def force_remove_readonly(func, path, _):
    """Clear the read-only flag and retry deletion."""
    os.chmod(path, stat.S_IWRITE)  # Make file writable
    func(path)  # Retry deletion

def delete_directory(repo_path):
    try:
        if not os.path.exists(repo_path):
            logging.info(f"Directory '{repo_path}' does not exist.")
            return False
        else:
            shutil.rmtree(repo_path, onerror=force_remove_readonly)
            logging.info(f"Directory '{repo_path}' deleted.")
    except Exception as e:
        logging.error(f"Error deleting directory contents: {e}")
        return False

def add_git_metrics(df, platform):
    # Set up the Docker containers for the tools
    setup_tools_containers()

    # Ensure the necessary columns exist in the DataFrame
    columns = ["size", "license", "main_language", "language_distribution", "#commits", "#branches", "#contributors"]
    for column in columns:
        if column not in df.columns:
            df[column] = None

    for index, row in df.iterrows():
        owner, repo = row["owner"], row["repo"]
        try:
            repo_path = os.path.join(os.getcwd(), "temp")
            # Convert Windows path to Docker-compatible format
            if plat.system() == "Windows":
                repo_path = repo_path.replace("\\", "/")

            if clone_repository(owner, repo, platform):
                df.at[index, "size"] = get_repo_size(repo_path)
                df.at[index, "license"] = get_repo_license(repo_path)
                df.at[index, "main_language"], df.at[index, "language_distribution"] = get_language_distribution(repo_path)
                df.at[index, "#commits"] = get_commit_count(repo_path)
                df.at[index, "#branches"] = get_branch_count(repo_path)
                df.at[index, "#contributors"] = get_contributor_count(repo_path)
            delete_directory(repo_path)
        except Exception as e:
            logging.error(f"Error processing repository '{owner}/{repo}': {e}")