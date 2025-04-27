import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from metrics import Metrics

# Define bins and labels for different metrics
metric_bins = {
    "commit": [1, 10, 10**2, 10**3, 10**4, 10**5, np.inf],
    "size": [1, 10**5, 10**6, 10**7, 10**8, 10**9, np.inf],
    "branch": [0, 1, 5, 20, 100, np.inf],
    "contributor": [0, 1, 5, 10, 20, np.inf],
    "issue": [-np.inf, 0, 5, 10, 100, np.inf],
    "pull_request": [-np.inf, 0, 5, 10, 100, np.inf],
    "updated": [0, 1, 7, 30, 180, 360, np.inf],
}

metric_labels = {
    "commit": ["1-10", "11-100", "101-1K", "1K-10K", "10K-100K", ">100K"],
    "size": ["1B-100KB","100KB-1MB", "1MB-10MB", "10MB-100MB", "100MB-1GB", ">1GB"],
    "branch": ["1", "2-5", "6-20", "21-100", ">100"],
    "contributor": ["1", "2-5", "6-10", "11-20", ">20"],
    "issue": ["0", "1-5", "6-10", "11-100", ">100"],
    "pull_request": ["0", "1-5", "6-10", "11-100", ">100"],
    "updated": ["<1 day", "1-7 days", "8-30 days", "31-180 days", "181-360 days", ">360 days"],
}

matching_features = {
    "git": ["#commits", "#branches", "#contributors", "size"],
    "platform": ["#stars", "#forks", "#issues", "#pull_requests"],
}

def get_clone_error_number(df_github, df_gitlab, df_gitea, df_forgejo):
    print(f"Github repositories that couldn't be cloned: {df_github['size'].isnull().sum()}")
    print(f"Gitlab repositories that couldn't be cloned: {df_gitlab['size'].isnull().sum()}")
    #print(f"Bitbucket repositories that couldn't be cloned: {df_bitbucket['size'].isnull().sum()}")
    print(f"Gitea repositories that couldn't be cloned: {df_gitea['size'].isnull().sum()}")
    print(f"Forgejo repositories that couldn't be cloned: {df_forgejo['size'].isnull().sum()}")

def get_most_present_owner(df_github, df_gitlab, df_gitea, df_forgejo):
    print(f"Most present owner in GitHub: {df_github['owner'].value_counts().idxmax()} has {df_github['owner'].value_counts().max()} repositories")
    print(f"Most present owner in Gitlab: {df_gitlab['owner'].value_counts().idxmax()} has {df_gitlab['owner'].value_counts().max()} repositories")
    #print(f"Most present owner in Bitbucket: {df_bitbucket['owner'].value_counts().idxmax()} has {df_bitbucket['owner'].value_counts().max()} repositories")
    print(f"Most present owner in Gitea: {df_gitea['owner'].value_counts().idxmax()} has {df_gitea['owner'].value_counts().max()} repositories")
    print(f"Most present owner in Forgejo: {df_forgejo['owner'].value_counts().idxmax()} has {df_forgejo['owner'].value_counts().max()} repositories")

def get_unique_owner_number(df_github, df_gitlab, df_gitea, df_forgejo):
    print(f"Unique owners in GitHub: {df_github['owner'].nunique()}")
    print(f"Unique owners in Gitlab: {df_gitlab['owner'].nunique()}")
    #print(f"Unique owners in Bitbucket: {df_bitbucket['owner'].nunique()}")
    print(f"Unique owners in Gitea: {df_gitea['owner'].nunique()}")
    print(f"Unique owners in Forgejo: {df_forgejo['owner'].nunique()}")

def count_languages(row):
    if pd.isna(row):
        return 0
    lang_dict = json.loads(row)
    return len(lang_dict)

def extract_major_languages(lang_json, threshold=20):
    """Extracts languages contributing more than a given threshold (default: 20%)."""
    try:
        lang_dict = json.loads(lang_json)
        return [lang for lang, details in lang_dict.items() if float(details.get("percentage", 0)) > threshold]
    except (json.JSONDecodeError, KeyError, ValueError):
        return ["No metric"]

def bin_alphanumeric_data(df, platform, metric, n_boot=1000):
    """
    Bootstraps categorical distributions for an alphanumeric metric (e.g., programming languages or licenses).
    Returns multiple bootstrap samples to enable error estimation.
    """
    metric_column = metric.value

    # Handle missing values
    df[metric_column] = df[metric_column].fillna("No metric")

    # Extract language data if the metric is LANGUAGE_DISTRIBUTION
    if metric == Metrics.LANGUAGE_DISTRIBUTION:
        categories = df[metric_column].apply(extract_major_languages).explode()
    else:
        categories = df[metric_column]

    bootstrapped_data = []

    for _ in range(n_boot):
        sample = categories.sample(frac=1, replace=True)  # Bootstrap resampling
        category_counts = sample.value_counts(normalize=True) * 100  # Convert counts to percentages

        for category, percentage in category_counts.items():
            bootstrapped_data.append({
                "platform": platform,
                metric_column: category,
                "percentage": percentage
            })

    return pd.DataFrame(bootstrapped_data)

def plot_alphanumeric_distribution(metric, df_github, df_gitlab, df_bitbucket, df_gitea, df_forgejo, n_boot=1000):
    """
    Plots the distribution of an alphanumeric metric (e.g., languages, licenses) across platforms using bootstrapped confidence intervals.
    """
    metric_name = metric.value

    df_github_bins = bin_alphanumeric_data(df_github, "GitHub", metric, n_boot)
    df_gitlab_bins = bin_alphanumeric_data(df_gitlab, "Gitlab", metric, n_boot)
    df_gitea_bins = bin_alphanumeric_data(df_gitea, "Gitea", metric, n_boot)
    df_forgejo_bins = bin_alphanumeric_data(df_forgejo, "Forgejo", metric, n_boot)

    if metric != Metrics.ISSUE:
        df_bitbucket_bins = bin_alphanumeric_data(df_bitbucket, "Bitbucket", metric, n_boot)
        df_bins = pd.concat([df_github_bins, df_gitlab_bins, df_bitbucket_bins, df_gitea_bins, df_forgejo_bins])
    else:
        df_bins = pd.concat([df_github_bins, df_gitlab_bins, df_gitea_bins, df_forgejo_bins])

    # Identify top 10 categories across platforms
    top_10_metric = df_bins.groupby(metric_name)["percentage"].mean().nlargest(10).index

    # Filter data for top categories
    df_bins = df_bins[df_bins[metric_name].isin(top_10_metric)]

    # Plot with bootstrapped confidence intervals
    plt.figure(figsize=(12, 8))
    sns.barplot(
        x=metric_name, y="percentage", hue="platform", data=df_bins,
        errorbar="sd", palette="tab10"
    )

    plt.title(f"{metric_name} Distribution on the Main Platforms (with Bootstrapped CI)")
    plt.xlabel(metric_name)
    plt.ylabel("Percentage of Repositories (%)")
    plt.xticks(rotation=45)
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.show()

    df_github_clean = df_github.copy()
    df_gitlab_clean = df_gitlab.copy()
    df_bitbucket_clean = df_bitbucket.copy()
    df_gitea_clean = df_gitea.copy()
    df_forgejo_clean = df_forgejo.copy()

    if metric == Metrics.MAIN_LANGUAGE:
        df_github_clean["num_languages"] = df_github_clean["language_distribution"].apply(count_languages)
        df_gitlab_clean["num_languages"] = df_gitlab_clean["language_distribution"].apply(count_languages)
        df_bitbucket_clean["num_languages"] = df_bitbucket_clean["language_distribution"].apply(count_languages)
        df_gitea_clean["num_languages"] = df_gitea_clean["language_distribution"].apply(count_languages)
        df_forgejo_clean["num_languages"] = df_forgejo_clean["language_distribution"].apply(count_languages)

        # Print max, mean, and median number of languages for each DataFrame
        print(f"GitHub - Max number of languages: {df_github_clean['num_languages'].max()}, Mean number of languages: {df_github_clean['num_languages'].mean():.2f}, Median number of languages: {df_github_clean['num_languages'].median()}")
        print(f"GitLab - Max number of languages: {df_gitlab_clean['num_languages'].max()}, Mean number of languages: {df_gitlab_clean['num_languages'].mean():.2f}, Median number of languages: {df_gitlab_clean['num_languages'].median()}")
        print(f"Bitbucket - Max number of languages: {df_bitbucket_clean['num_languages'].max()}, Mean number of languages: {df_bitbucket_clean['num_languages'].mean():.2f}, Median number of languages: {df_bitbucket_clean['num_languages'].median()}")
        print(f"Gitea - Max number of languages: {df_gitea_clean['num_languages'].max()}, Mean number of languages: {df_gitea_clean['num_languages'].mean():.2f}, Median number of languages: {df_gitea_clean['num_languages'].median()}")
        print(f"Forgejo - Max number of languages: {df_forgejo_clean['num_languages'].max()}, Mean number of languages: {df_forgejo_clean['num_languages'].mean():.2f}, Median number of languages: {df_forgejo_clean['num_languages'].median()}")

    if metric == Metrics.LICENSE:
        github_counts = df_github[metric.value].value_counts().get('No metric', 0)
        gitlab_counts = df_gitlab[metric.value].value_counts().get('No metric', 0)
        bitbucket_counts = df_bitbucket[metric.value].value_counts().get('No metric', 0)
        gitea_counts = df_gitea[metric.value].value_counts().get('No metric', 0)
        forgejo_counts = df_forgejo[metric.value].value_counts().get('No metric', 0)
        print(f"GitHub repositories that have a license: {((len(df_github) - github_counts) / len(df_github)) * 100:.2f}%")
        print(f"Gitlab repositories that have a license: {((len(df_gitlab) - gitlab_counts) / len(df_gitlab)) * 100:.2f}%")
        print(f"Bitbucket repositories that have a license: {((len(df_bitbucket) - bitbucket_counts) / len(df_bitbucket)) * 100:.2f}%")
        print(f"Gitea repositories that have a license: {((len(df_gitea) - gitea_counts) / len(df_gitea)) * 100:.2f}%")
        print(f"Forgejo repositories that have a license: {((len(df_forgejo) - forgejo_counts) / len(df_forgejo)) * 100:.2f}%")



def bin_data(df, platform, metric, n_boot):
    """
    Bin the data based on the given metric and perform bootstrapping to estimate confidence intervals.
    Returns a DataFrame with multiple bootstrap samples for each platform.
    """
    metric_name = str(metric.name.lower())
    metric_column = metric.value

    df = df.dropna(subset=[metric_column]).copy()

    # Check if the metric column is a date
    if metric == Metrics.UPDATED:
        df[metric_column] = pd.to_datetime(df[metric_column], errors='coerce')
        df["created"] = pd.to_datetime(df["created"], errors='coerce')
        df["updated"] = (df[metric_column] - df["created"]).dt.total_seconds() / (60 * 60 * 24)  # Convert to days
        df[metric_column] = df["updated"]

    df.loc[:, metric_column] = pd.to_numeric(df[metric_column], errors='coerce')

    # Define bins based on the metric
    bins = metric_bins.get(metric_name)
    labels = metric_labels.get(metric_name)

    df[f"{metric_name}_range"] = pd.cut(df[metric_column], bins=bins, labels=labels, include_lowest=True)

    # Bootstrapping: Generate multiple samples and calculate percentages
    bootstrapped_data = []

    for _ in range(n_boot):
        sample = df.sample(frac=1, replace=True)  # Bootstrap resampling
        bin_counts = sample[f"{metric_name}_range"].value_counts().reindex(labels, fill_value=0)
        total_repos = bin_counts.sum()
        bin_percentages = (bin_counts / total_repos) * 100  # Convert to percentage

        for bin_label, percentage in bin_percentages.items():
            bootstrapped_data.append({
                "platform": platform,
                f"{metric_name}_range": bin_label,
                "percentage": percentage
            })
    return pd.DataFrame(bootstrapped_data)


def plot_numeric_distribution(df_github, df_gitlab, df_bitbucket, df_gitea, df_forgejo, metric, n_boot=1000):
    """
    Plot the distribution of a given metric across different platforms with bootstrapped confidence intervals.
    """
    metric_name = str(metric.name.lower())

    df_github_bins = bin_data(df_github, "GitHub", metric, n_boot)
    df_gitlab_bins = bin_data(df_gitlab, "Gitlab", metric, n_boot)
    df_gitea_bins = bin_data(df_gitea, "Gitea", metric, n_boot)
    df_forgejo_bins = bin_data(df_forgejo, "Forgejo", metric, n_boot)

    if metric != Metrics.ISSUE:
        df_bitbucket_bins = bin_data(df_bitbucket, "Bitbucket", metric, n_boot)
        df_bins = pd.concat([df_github_bins, df_gitlab_bins, df_bitbucket_bins, df_gitea_bins, df_forgejo_bins])
    else:
        df_bins = pd.concat([df_github_bins, df_gitlab_bins, df_gitea_bins, df_forgejo_bins])

    # Plot using the bootstrapped data
    plt.figure(figsize=(12, 6))
    sns.barplot(
        x=f"{metric_name}_range",
        y="percentage",
        hue="platform",
        data=df_bins,
        errorbar="sd",  # Since we already bootstrapped, we show the spread using standard deviation
        palette="tab10"
    )

    plt.xlabel(f"{metric_name} ranges")
    plt.ylabel("Percentage of Repositories (%)")
    plt.title(f"Repository {metric_name} Distribution Across Platforms with Bootstrapped CI")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(title="Platform")

    plt.show()

    if metric == Metrics.UPDATED:
        # Convert date columns to Unix timestamps
        df_github[metric.value] = (pd.to_datetime(df_github[metric.value], errors='coerce').astype(int) / 10 ** 9 -
                                   pd.to_datetime(df_github["created"], errors='coerce').astype(int) / 10 ** 9) / (60 * 60 * 24)
        df_gitlab[metric.value] = (pd.to_datetime(df_gitlab[metric.value], errors='coerce').astype(int) / 10 ** 9 -
                                   pd.to_datetime(df_gitlab["created"], errors='coerce').astype(int) / 10 ** 9) / (60 * 60 * 24)
        df_bitbucket[metric.value] = (pd.to_datetime(df_bitbucket[metric.value], errors='coerce').astype(int) / 10 ** 9 -
                                   pd.to_datetime(df_bitbucket["created"], errors='coerce').astype(int) / 10 ** 9) / (60 * 60 * 24)
        df_gitea[metric.value] = (pd.to_datetime(df_gitea[metric.value], errors='coerce').astype(int) / 10 ** 9 -
                                   pd.to_datetime(df_gitea["created"], errors='coerce').astype(int) / 10 ** 9) / (60 * 60 * 24)
        df_forgejo[metric.value] = (pd.to_datetime(df_forgejo[metric.value], errors='coerce').astype(int) / 10 ** 9 -
                                   pd.to_datetime(df_forgejo["created"], errors='coerce').astype(int) / 10 ** 9) / (60 * 60 * 24)

    # Print max, mean, and median values for each DataFrame
    print(f"GitHub - Max {metric_name}: {int(df_github[metric.value].max())}, Mean {metric_name}: {int(df_github[metric.value].mean())}, Median {metric_name}: {int(df_github[metric.value].median())}")
    print(f"GitLab - Max {metric_name}: {int(df_gitlab[metric.value].max())}, Mean {metric_name}: {int(df_gitlab[metric.value].mean())}, Median {metric_name}: {int(df_gitlab[metric.value].median())}")
    if metric != Metrics.ISSUE:
        print(f"Bitbucket - Max {metric_name}: {int(df_bitbucket[metric.value].max())}, Mean {metric_name}: {int(df_bitbucket[metric.value].mean())}, Median {metric_name}: {int(df_bitbucket[metric.value].median())}")
    print(f"Gitea - Max {metric_name}: {int(df_gitea[metric.value].max())}, Mean {metric_name}: {int(df_gitea[metric.value].mean())}, Median {metric_name}: {int(df_gitea[metric.value].median())}")
    print(f"Forgejo - Max {metric_name}: {int(df_forgejo[metric.value].max())}, Mean {metric_name}: {int(df_forgejo[metric.value].mean())}, Median {metric_name}: {int(df_forgejo[metric.value].median())}")

def plot_step_lines(df_github, df_gitlab, df_gitea, df_forgejo):
    plt.figure(figsize=(12, 6))

    # Dictionary to store dataframes and their platform labels
    platforms = {
        "GitHub": df_github,
        "GitLab": df_gitlab,
        #"Bitbucket": df_bitbucket,
        "Gitea": df_gitea,
        "Forgejo": df_forgejo
    }

    # Create a new DataFrame to store all data
    combined_data = []

    for label, df in platforms.items():
        # Convert to datetime and extract date
        df["created"] = pd.to_datetime(df["created"])
        df["date"] = df["created"].dt.date

        # Count occurrences per day
        date_counts = df["date"].value_counts().sort_index().cumsum()

        # Append platform name as a column
        temp_df = pd.DataFrame({"date": date_counts.index, "cumulative_count": date_counts.values, "Platform": label})
        combined_data.append(temp_df)

    # Combine all platform data
    combined_df = pd.concat(combined_data)

    # Plot using Seaborn with hue for automatic color assignment
    sns.lineplot(data=combined_df, x="date", y="cumulative_count", hue="Platform", drawstyle="steps-mid", marker="o")

    # Formatting
    plt.xlabel("Date")
    plt.ylabel("Cumulative Count")
    plt.title("Cumulative Repositories Created Over Time")
    plt.xticks(rotation=45)
    plt.legend(title="Platform")
    plt.grid()

    plt.show()


def create_correlation_matrix(df):
    """
    Create a correlation matrix for repository metrics and plot it.
    """
    numeric_columns = ["created", "updated", "#stars", "#forks", "size",
                       "#commits", "#branches", "#contributors", "#issues", "#pull_requests"]

    # Only keep columns that exist
    existing_columns = [col for col in numeric_columns if col in df.columns]

    # Convert datetime columns before subsetting
    if 'created' in df.columns:
        df['created'] = pd.to_datetime(df['created'], errors='coerce')
    if 'updated' in df.columns:
        df['updated'] = pd.to_datetime(df['updated'], errors='coerce')

    df_subset = df[existing_columns].copy()

    # Convert all numeric columns to float (except datetime)
    for col in df_subset.select_dtypes(include=['object']).columns:
        df_subset[col] = pd.to_numeric(df_subset[col], errors='coerce')

    # Add derived feature: age_days
    if 'created' in df.columns and 'updated' in df.columns:
        df_subset['age_days'] = (df['updated'] - df['created']).dt.total_seconds() / (60 * 60 * 24)

    # Compute and plot correlation matrix
    corr_matrix = df_subset.corr(numeric_only=True)

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation Matrix of Repository Metrics")
    plt.show()


def plot_lorenz_curve(values, title="Lorenz Curve"):
    # Drop missing values and sort
    values = np.array(sorted(values.dropna()))

    # Cumulative share of values
    cumulative_values = np.cumsum(values)
    cumulative_share = cumulative_values / cumulative_values[-1]

    # Cumulative share of population (uniform steps)
    n = len(values)
    cumulative_population = np.arange(1, n + 1) / n

    # Add starting point (0, 0)
    cumulative_share = np.insert(cumulative_share, 0, 0)
    cumulative_population = np.insert(cumulative_population, 0, 0)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(cumulative_population, cumulative_share, label='Lorenz Curve')
    plt.plot([0, 1], [0, 1], '--', color='gray', label='Equality Line')
    plt.title(title)
    plt.xlabel('Cumulative Share of Repositories')
    plt.ylabel('Cumulative Share of Commits')
    plt.legend()
    plt.grid(True)
    plt.show()


def propensity_score_matching(df_platform1, df_platform2, metric, scale="linear", max_difference=0.2):
    """
    Perform propensity score matching based on a given metric and plot the matched repository sizes.
    :param df_platform1: DataFrame containing repositories from platform 1.
    :param df_platform2: DataFrame containing repositories from platform 2.
    :param metric: Metric to compare.
    :param max_difference: Maximum difference in propensity scores for matching.
    :return: A scatter plot showing the matched repository metric repartition.
    """
    df_platform1["platform"] = 1  # Treatment
    df_platform2["platform"] = 0  # Control
    df = pd.concat([df_platform1, df_platform2])

    # Use common metrics for matching but exclude the metric we want to compare
    if metric.value in matching_features["git"]:
        features = matching_features["git"].copy()
        features.remove(metric.value)
    else:
        features = matching_features["platform"].copy()
        features.remove(metric.value)
    print(features)
    df = df.dropna(subset=features)

    # Normalize Features
    scaler = StandardScaler()
    x = scaler.fit_transform(df[features])
    y = df["platform"]

    # Fit Logistic Regression for Propensity Scores
    logit = LogisticRegression()
    logit.fit(x, y)

    df["propensity_score"] = logit.predict_proba(x)[:, 1]

    # Separate Groups
    df_treated = df[df["platform"] == 1].copy()
    df_control = df[df["platform"] == 0].copy()

    # Matching Using Nearest Neighbors
    x_treated = df_treated["propensity_score"].values.reshape(-1, 1)
    x_control = df_control["propensity_score"].values.reshape(-1, 1)

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(x_treated)
    distances, indices = nn.kneighbors(x_control, n_neighbors=1)

    # Compute Propensity Score Differences
    matched_indices = indices.flatten()
    propensity_differences = np.abs(df_treated.iloc[matched_indices]["propensity_score"].values - df_control["propensity_score"].values)

    # Filter by max difference threshold
    valid_matches = propensity_differences <= max_difference
    print(f"Number of valid matches: {valid_matches.sum()} / {len(propensity_differences)}")

    df_matched_treated = df_treated.iloc[matched_indices[valid_matches]].reset_index(drop=True)
    df_matched_control = df_control[valid_matches].reset_index(drop=True)

    # Get Actual Sizes for Matched Pairs
    github_sizes = df_matched_treated[metric.value].values
    gitea_sizes = df_matched_control[metric.value].values

    # Ensure Both Axes Use the Same Scale
    max_size = max(github_sizes.max(), gitea_sizes.max())

    # Set a small buffer to avoid log scale errors with zero values
    max_size = max_size * 1.1  # Add a 10% buffer

    # Create the Plot
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        x=github_sizes,
        y=gitea_sizes,
        alpha=0.7,
        label="Matched Repositories"  # Label for legend
    )

    # Add y = x line (Perfect Match Reference)
    plt.plot([0, max_size], [0, max_size], color="red", linestyle="--", label="Perfect Match (y = x)")
    plt.xlabel(f"GitHub {metric.value}")
    plt.ylabel(f"Gitea {metric.value}")
    plt.xscale(scale)
    plt.xticks(rotation=90)
    plt.yscale(scale)
    plt.yticks(rotation=90)
    plt.xlim(None, max_size)
    plt.ylim(None, max_size)
    plt.title(f"Matched Repository Sizes: GitHub vs Gitea (Max Difference ≤ {max_difference})")
    plt.legend()

    plt.show()
