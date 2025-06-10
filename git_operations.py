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
import time

# Set up logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

def setup_tools_containers():
    """Set up Docker containers with GitHub-Linguist and Licensee installed."""
    git_tools.check_docker()
    git_tools.build_licensee_image()
    git_tools.build_linguist_image()

def clone_repository(owner, repo, platform, repo_path, shallow=False, metadata=False):
    """
    Function to clone a repository from a platform-specific API.
    :param owner: Owner of the repository.
    :param repo: Repository name.
    :param platform: Platform to fetch the repository from.
    :param repo_path: Local path where the repository will be cloned.
    :param shallow: If True, performs a shallow clone (only the latest commit).
    :param metadata: If True, performs a metadata-only clone (no files checked out).
    """
    temp = platform.name + "_CLONE"
    url = getattr(Endpoints, temp)(owner, repo)
    try:
        if os.path.exists(repo_path):
            shutil.rmtree(repo_path)
        logging.info(f"Cloning {url} into {repo_path}...")
        if shallow:
            git.Repo.clone_from(url, repo_path, depth=1)
        elif metadata:
            git.Repo.clone_from(url, repo_path, filter="blob:none", no_checkout=True)
        else:
            git.Repo.clone_from(url, repo_path)
        logging.info("Clone successful!")
        return True

    except Exception as e:
        logging.error(f"Error cloning repository: {e}")
        raise

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
        return None, None

def get_commit_count(repo_path):
    """Get the number of commits in a Git repository."""
    try:
        result = subprocess.run(
            ["git", "rev-list", "--count", "HEAD"],
            cwd=repo_path,  # Run the command inside the repo
            capture_output=True,
            text=True,
            check=True
        )

        commit_count = int(result.stdout.strip())
        return commit_count

    except subprocess.CalledProcessError as e:
        logging.error(f"Error getting commit count: {e}")
        return None

def get_branch_count(repo_path):
    """Get the number of branches in a Git repository."""
    try:
        result = subprocess.run(
            ["git", "branch", "-a"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )

        branches = result.stdout.strip().split("\n")
        branch_count = len([branch for branch in branches if branch.strip()]) #-2 to avoid counting current and head
        if branch_count > 2: #Empty repository case
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

def add_git_metrics(df, platform, path, delete_repositories=True):
    """Function to add various Git metrics to a DataFrame of repositories.
    :param df: DataFrame containing repository information.
    :param platform: Platform to fetch the repositories from.
    :param path: Path where the repositories will be cloned.
    :param delete_repositories: If True, delete the cloned repositories after processing.
    """
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
            repo_path = os.path.join(path, platform.name, f"{owner}-{repo}")
            # Convert Windows path to Docker-compatible format
            if plat.system() == "Windows":
                repo_path = repo_path.replace("\\", "/")

            if clone_repository(owner, repo, platform, repo_path, metadata=True):
                df.at[index, "#commits"] = get_commit_count(repo_path)
                df.at[index, "#branches"] = get_branch_count(repo_path)
                df.at[index, "#contributors"] = get_contributor_count(repo_path)
            if clone_repository(owner, repo, platform, repo_path, shallow=True):
                df.at[index, "size"] = get_repo_size(repo_path)
                df.at[index, "license"] = get_repo_license(repo_path)
                df.at[index, "main_language"], df.at[index, "language_distribution"] = get_language_distribution(repo_path)
            if delete_repositories:
                delete_directory(repo_path)
        except Exception as e:
            logging.error(f"Error processing repository '{owner}/{repo}': {e}")
            raise

def compare_git_clone_speed(df, platform):
    """Function to compare the speed of shallow and full cloning of repositories.
    :param df: DataFrame containing repository information.
    :param platform: Platform to fetch the repositories from.
    """

    # Ensure the necessary columns exist in the DataFrame
    columns = ["shallow_clone_time", "full_clone_time"]
    for column in columns:
        if column not in df.columns:
            df[column] = None

    for index, row in df.iterrows():
        owner, repo = row["owner"], row["repo"]
        logging.info(f"Processing repository {index + 1} out of {len(df)}")
        try:
            repo_path = os.path.join(os.getcwd(), "temp")
            # Convert Windows path to Docker-compatible format
            if plat.system() == "Windows":
                repo_path = repo_path.replace("\\", "/")

            # Shallow clone
            start_time = time.time()
            clone_repository(owner, repo, platform, repo_path, metadata=True)
            clone_repository(owner, repo, platform, repo_path, shallow=True)
            shallow_clone_time = time.time() - start_time
            df.at[index, "shallow_clone_time"] = shallow_clone_time

            # Full clone
            start_time = time.time()
            clone_repository(owner, repo, platform, repo_path)
            full_clone_time = time.time() - start_time
            df.at[index, "full_clone_time"] = full_clone_time

            delete_directory(repo_path)
        except Exception as e:
            logging.error(f"Error processing repository '{owner}/{repo}': {e}")