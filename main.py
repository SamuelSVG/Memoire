import time

import requests
import pandas as pd
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Load environment variables from .env file
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITLAB_TOKEN = os.getenv("GITLAB_TOKEN")
GITEA_TOKEN = os.getenv("GITEA_TOKEN")
CODEBERG_TOKEN = os.getenv("CODEBERG_TOKEN")
BITBUCKET_TOKEN = os.getenv("BITBUCKET_TOKEN")

# Function to fetch repositories from ecosyste.ms API
def fetch_repos_from_ecosystems(platform, num_repos=100):
    repo_url = f"https://repos.ecosyste.ms/api/v1/hosts/{platform}/repositories"
    page = 1
    params = {"page": page, "per_page": 50}
    repos = []

    while len(repos) < num_repos:

        response = requests.get(repo_url, params=params)
        if response.status_code != 200:
            print(f"Error fetching from ecosyste.ms: {response.status_code}")
            break

        params["page"] += 1
        data = response.json()
        valid_repos = [repo for repo in data if repo.get("size") and repo["size"] > 0]
        repos.extend(valid_repos)

        if len(valid_repos) < len(data):  # If there are invalid repos, continue fetching
            print("Some repositories were skipped due to size=0.")

        if len(data) < 50:  # If fewer than 50 repos were returned, stop to prevent infinite looping
            break

    return repos


# Function to fetch repository details from GitHub API
def fetch_github_repo_data(owner, repo, token):
    headers = {"Authorization": f"Bearer {token}"}
    repo_url = f"https://api.github.com/repos/{owner}/{repo}"

    response = requests.get(repo_url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        return {
            "Platform": "GitHub",
            "Name": data.get("name"),
            "Stars": data.get("stargazers_count"),
            "Forks": data.get("forks_count"),
            "Open Issues": data.get("open_issues_count"),
            "Created At": data.get("created_at"),
            "Updated At": data.get("updated_at"),
            "Size": data.get("size"),
            "Language": data.get("language")
        }
    else:
        print(f"Error fetching GitHub data for {owner}/{repo}: {response.status_code}")
        return None


# Function to fetch repository details from GitLab API
def fetch_gitlab_repo_data(project_id, token):
    headers = {"Authorization": f"Bearer {token}"}
    repo_url = f"https://gitlab.com/api/v4/projects/{project_id}"

    response = requests.get(repo_url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        return {
            "Platform": "GitLab",
            "Name": data.get("name"),
            "Stars": data.get("star_count"),
            "Forks": data.get("forks_count"),
            "Open Issues": data.get("open_issues_count"),
            "Created At": data.get("created_at"),
            "Updated At": data.get("last_activity_at"),
            "Size": None, # GitLab does not directly provide size info,
            "Language": None  # GitLab does not directly provide language info
        }
    else:
        print(f"Error fetching GitLab data for {project_id}: {response.status_code}")
        return None

# Function to fetch repository details from Gitea API
def fetch_gitea_repo_data(owner, repo, token):
    headers = {"Authorization": f"token {token}"}
    repo_url = f"https://gitea.com/api/v1/repos/{owner}/{repo}"

    response = requests.get(repo_url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        return {
            "Platform": "Gitea",
            "Name": data.get("name"),
            "Stars": data.get("stars_count"),
            "Forks": data.get("forks_count"),
            "Open Issues": data.get("open_issues_count"),
            "Created At": data.get("created_at"),
            "Updated At": data.get("updated_at"),
            "Size": data.get("size"),
            "Language": data.get("language")
        }
    else:
        print(f"Error fetching Gitea data for {owner}/{repo}: {response.status_code}")
        return None

def fetch_codeberg_repo_data(owner, repo, token):
    headers = {"Authorization": f"token {token}"}
    repo_url = f"https://codeberg.org/api/v1/repos/{owner}/{repo}"

    response = requests.get(repo_url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        return {
            "Platform": "Codeberg",
            "Name": data.get("name"),
            "Stars": data.get("stars_count"),
            "Forks": data.get("forks_count"),
            "Open Issues": data.get("open_issues_count"),
            "Created At": data.get("created_at"),
            "Updated At": data.get("updated_at"),
            "Size": data.get("size"),
            "Language": data.get("language")
        }
    else:
        print(f"Error fetching Gitea data for {owner}/{repo}: {response.status_code}")
        return None

def fetch_bitbucket_repo_data(owner, repo, token):
    headers = {"Authorization": f"Bearer {token}"}
    repo_url = f"https://api.bitbucket.org/2.0/repositories/{owner}/{repo}"

    response = requests.get(repo_url)
    if response.status_code == 200:
        data = response.json()
        repo_info = {
            "Platform": "Bitbucket",
            "Name": data['name'],
            "Stars": data.get('stars', None),  # TODO Check the correct endpoint
            "Forks": data.get('fork_policy', None),  # Bitbucket API returns fork_policy, may need interpretation
            "Open Issues": None,  # Requires separate issues API endpoint
            "Created At": data['created_on'],
            "Updated At": data['updated_on']
        }
        return repo_info
    else:
        print(f"Error fetching Bitbucket repo data: {response.status_code}, {response.json()}")
        return None


# Combine data for repositories fetched from ecosyste.ms
def fetch_and_store_data(github_token, gitlab_token, gitea_token, codeberg_token):
    combined_data = []

    # Fetch GitHub repositories
    github_repos = fetch_repos_from_ecosystems(platform="Github")
    for repo in github_repos:
        full_name = repo.get("full_name", "")
        if "/" in full_name:
            owner, repo_name = full_name.split("/", 1)
            github_data = fetch_github_repo_data(owner, repo_name, github_token)
            if github_data:
                combined_data.append(github_data)

    """
    # Fetch GitLab repositories
    gitlab_repos = fetch_repos_from_ecosystems(platform="Gitlab.com")
    request_counter = 0
    for repo in gitlab_repos:
        if request_counter >= 50:
            print("I am sleeping")
            time.sleep(60)
        project_id = repo.get("uuid", "")
        gitlab_data = fetch_gitlab_repo_data(project_id, gitlab_token)
        request_counter += 1
        if gitlab_data:
            combined_data.append(gitlab_data)
    """

    # Fetch Gitea repositories
    gitea_repos = fetch_repos_from_ecosystems(platform="Gitea.com")
    request_counter = 0
    for repo in gitea_repos:
        if request_counter >= 50:
            print("I am sleeping")
            time.sleep(60)
        full_name = repo.get("full_name", "")
        if "/" in full_name:
            owner, repo_name = full_name.split("/", 1)
            request_counter += 1
            gitea_data = fetch_gitea_repo_data(owner, repo_name, gitea_token)
            if gitea_data:
                combined_data.append(gitea_data)

    # Fetch Codeberg repositories
    codeberg_repos = fetch_repos_from_ecosystems(platform="Codeberg.org")
    request_counter = 0
    for repo in codeberg_repos:
        if request_counter >= 50:
            print("I am sleeping")
            time.sleep(60)
        full_name = repo.get("full_name", "")
        if "/" in full_name:
            owner, repo_name = full_name.split("/", 1)
            request_counter += 1
            codeberg_data = fetch_codeberg_repo_data(owner, repo_name, codeberg_token)
            if codeberg_data:
                combined_data.append(codeberg_data)
    """
    # Fetch Bitbucket repositories
    bitbucket_repos = fetch_repos_from_ecosystems(platform="Bitbucket.org")
    for repo in bitbucket_repos:
        full_name = repo.get("full_name", "")
        if "/" in full_name:
            owner, repo_name = full_name.split("/", 1)
            bitbucket_data = fetch_bitbucket_repo_data(owner, repo_name, codeberg_token)
            if bitbucket_data:
                combined_data.append(bitbucket_data)
    """

    # Create a DataFrame
    df = pd.DataFrame(combined_data)
    return df

def first_plot():
    # Load the CSV file
    csv_file = "combined_repository_data.csv"
    df = pd.read_csv(csv_file)

    # Group by Platform and Language, and count the number of repositories
    grouped_data = (
        df.groupby(["Platform", "Language"])
        .size()
        .reset_index(name="Repository Count")
    )

    # Get the list of unique platforms
    platforms = grouped_data["Platform"].unique()

    # Create separate plots for each platform
    plt.figure(figsize=(12, len(platforms) * 5))  # Adjust the figure size based on the number of platforms
    for i, platform in enumerate(platforms):
        platform_data = grouped_data[grouped_data["Platform"] == platform]
        platform_data = platform_data.sort_values("Repository Count", ascending=False)  # Sort for clarity

        plt.subplot(len(platforms), 1, i + 1)  # Create a subplot for each platform
        sns.barplot(
            x="Repository Count",
            y="Language",
            data=platform_data,
            palette="Blues_d",
        )

        plt.title(f"Number of Repositories per Language on {platform}", fontsize=14)
        plt.xlabel("Repository Count", fontsize=12)
        plt.ylabel("Programming Language", fontsize=12)

        # Adjust tick size for better readability
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    """
    # Fetch and store combined data
    df = fetch_and_store_data(GITHUB_TOKEN, GITLAB_TOKEN, GITEA_TOKEN, CODEBERG_TOKEN)

    # Display the DataFrame
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print("Combined Repository DataFrame:")
        print(df)

    # Save to CSV
    df.to_csv("combined_repository_data.csv", index=False)
    print("Combined repository data saved to 'combined_repository_data.csv'")
    """

    first_plot()
