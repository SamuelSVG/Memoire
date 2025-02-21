import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from metrics import Metrics

# Define bins and labels for different metrics
metric_bins = {
    "commit": [1, 10, 100, 1000, 10000, 100000, 1000000],
    "size": [1, 10**3, 10**4, 10**5, 10**6, 10**7],
    "branch": [0, 1, 5, 20, 100, 500]
}

metric_labels = {
    "commit": ["1-10", "11-100", "101-1K", "1K-10K", "10K-100K", "100K-1M"],
    "size": ["1KB-1MB", "1MB-10MB", "10MB-100MB", "100MB-1GB", "1GB-10GB"],
    "branch": ["1", "2-5", "6-20", "21-100", "101-500"]
}

def plot_language_distribution(df_github, df_gitea):
    """
    Plot the distribution of programming languages across different platforms repositories.
    :param df_github: DataFrame containing GitHub repositories.
    :param df_gitea: DataFrame containing Gitea repositories.
    :return: A plot showing the distribution of programming languages.
    """
    # Remove missing values
    df_github = df_github.dropna(subset=['language'])
    df_gitea = df_gitea.dropna(subset=['language'])

    # Count repositories per language
    github_counts = df_github['language'].value_counts()
    gitea_counts = df_gitea['language'].value_counts()

    # Calculate the percentage of repositories for each language
    github_percentages = (github_counts / github_counts.sum()) * 100
    gitea_percentages = (gitea_counts / gitea_counts.sum()) * 100

    # Identify top 10 languages across both platforms
    top_languages = set(github_percentages.head(10).index) | set(gitea_percentages.head(10).index)

    # Filter data to include only the top languages
    github_percentages = github_percentages[github_percentages.index.isin(top_languages)]
    gitea_percentages = gitea_percentages[gitea_percentages.index.isin(top_languages)]

    # Merge the data to ensure all top languages appear
    top_languages_df = pd.concat([github_percentages, gitea_percentages], axis=1, keys=['GitHub', 'Gitea']).fillna(0)
    top_languages_df = top_languages_df.reset_index()
    top_languages_df.columns = ['language', 'GitHub', 'Gitea']

    # Melt the data for plotting
    top_languages_df = top_languages_df.melt(id_vars='language', var_name='platform', value_name='percentage')

    # Plot the bar chart
    plt.figure(figsize=(12, 8))
    sns.barplot(x='language', y='percentage', hue='platform', data=top_languages_df, palette='tab10')
    plt.title("Language Distribution on GitHub and Gitea")
    plt.xlabel("Programming Language")
    plt.ylabel("Percentage of Repositories (%)")
    plt.xticks(rotation=45)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.show()

# TODO - Add the plot_language_distribution_sankey function
def plot_language_distribution_sankey(df_github, df_gitea):
    # Remove missing values
    df_github = df_github.dropna(subset=['Language'])
    df_gitea = df_gitea.dropna(subset=['Language'])

    # Count repositories per language
    github_counts = df_github['Language'].value_counts()
    gitea_counts = df_gitea['Language'].value_counts()

    # Identify top N languages across both platforms
    top_languages = set(github_counts.head(10).index) | set(gitea_counts.head(10).index)

    # Filter data for top languages
    github_counts = github_counts[github_counts.index.isin(top_languages)]
    gitea_counts = gitea_counts[gitea_counts.index.isin(top_languages)]

    # Merge into a DataFrame, ensuring all languages appear
    top_languages_df = pd.DataFrame(index=list(top_languages))
    top_languages_df['GitHub'] = github_counts
    top_languages_df['Gitea'] = gitea_counts
    top_languages_df = top_languages_df.fillna(0).reset_index().rename(columns={'index': 'Language'})

    # Create lists for Sankey diagram
    sources, targets, values = [], [], []
    labels = []
    node_index = {}
    idx = 0

    for _, row in top_languages_df.iterrows():
        language = row['Language']
        github_repos = row['GitHub']
        gitea_repos = row['Gitea']

        # Create GitHub node
        github_node = f"GitHub - {language}"
        if github_node not in node_index:
            node_index[github_node] = idx
            labels.append(github_node)
            idx += 1

        # Create Gitea node
        gitea_node = f"Gitea - {language}"
        if gitea_node not in node_index:
            node_index[gitea_node] = idx
            labels.append(gitea_node)
            idx += 1

        # Create flow from GitHub to Gitea
        sources.append(node_index[github_node])
        targets.append(node_index[gitea_node])
        values.append(min(github_repos, gitea_repos))  # Match correctly

    # Create Sankey Diagram
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values
        )
    ))

    fig.update_layout(title_text="Language Distribution on GitHub and Gitea (Sankey Diagram)", font_size=12)
    fig.show()


def bin_data(df, platform, metric):
    """
    Bin the data based on the given metric and calculate the percentage of repositories in each bin.
    :param df: DataFrame containing repository data.
    :param platform: Platform name.
    :param metric: Metric to bin the data.
    :return: DataFrame with binned data and percentage of repositories in each bin.
    """
    metric_name = str(metric.name.lower())
    metric_column = metric.value

    df = df.dropna(subset=[metric_column]).copy()
    df.loc[:, metric_column] = pd.to_numeric(df[metric_column], errors='coerce')

    # Define bins based on the metric
    bins = metric_bins.get(metric_name)
    labels = metric_labels.get(metric_name)

    # Create a new column with metric ranges
    df[f"{metric_name}_range"] = pd.cut(df[metric_column], bins=bins, labels=labels, include_lowest=True)

    # Count repositories per branch range
    metric_counts = df[f'{metric_name}_range'].value_counts().reindex(labels, fill_value=0)

    # Convert counts to percentages
    total_repos = metric_counts.sum()
    metric_percentages = (metric_counts / total_repos) * 100

    return pd.DataFrame({'platform': platform, f'{metric_name}_range': labels, 'percentage': metric_percentages.values})


def plot_distribution(df_github, df_gitea, metric):
    """
    Plot the distribution of a given metric across different platforms.
    :param df_github: DataFrame containing GitHub repositories.
    :param df_gitea: DataFrame containing Gitea repositories.
    :param metric: Metric to plot the distribution for.
    :return: A plot showing the distribution of the metric.
    """
    # Get the appropriate function based on the metric
    metric_name = str(metric.name.lower())

    df_github_bins = bin_data(df_github, "GitHub", metric)
    df_gitea_bins = bin_data(df_gitea, "Gitea", metric)
    df_bins = pd.concat([df_github_bins, df_gitea_bins])

    # Create the figure
    plt.figure(figsize=(12, 6))
    sns.barplot(x=f"{metric_name}_range", y="percentage", hue="platform", data=df_bins, palette="tab10")

    plt.xlabel(f"{metric_name} ranges")
    plt.ylabel("Percentage of Repositories (%)")
    plt.title(f"Repository {metric_name} Distribution Across Platforms")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(title="Platform")

    plt.show()


def propensity_score_matching(df_platform1, df_platform2, metric, max_difference=0.2):
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
    features = []
    for feature in Metrics:
        if feature != metric and feature != Metrics.LANGUAGE and feature != Metrics.LICENSE:
            features.append(feature.value)
    df = df.dropna(subset=features)

    # Normalize Features
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])
    y = df["platform"]

    # Fit Logistic Regression for Propensity Scores
    logit = LogisticRegression()
    logit.fit(X, y)

    df["propensity_score"] = logit.predict_proba(X)[:, 1]

    # Separate Groups
    df_treated = df[df["platform"] == 1].copy()
    df_control = df[df["platform"] == 0].copy()

    # Matching Using Nearest Neighbors
    X_treated = df_treated["propensity_score"].values.reshape(-1, 1)
    X_control = df_control["propensity_score"].values.reshape(-1, 1)

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(X_treated)
    distances, indices = nn.kneighbors(X_control, n_neighbors=1)

    # Compute Propensity Score Differences
    matched_indices = indices.flatten()
    propensity_differences = np.abs(df_treated.iloc[matched_indices]["propensity_score"].values - df_control["propensity_score"].values)

    # Filter by max difference threshold
    valid_matches = propensity_differences <= max_difference
    print(f"Number of valid matches: {valid_matches.sum()}")

    df_matched_treated = df_treated.iloc[matched_indices[valid_matches]].reset_index(drop=True)
    df_matched_control = df_control[valid_matches].reset_index(drop=True)

    # Get Actual Sizes for Matched Pairs
    github_sizes = df_matched_treated[metric.value].values
    gitea_sizes = df_matched_control[metric.value].values

    # Ensure Both Axes Use the Same Scale
    min_size = min(github_sizes.min(), gitea_sizes.min())
    max_size = max(github_sizes.max(), gitea_sizes.max())

    # Set a small buffer to avoid log scale errors with zero values
    min_size = max(min_size, 1)  # Ensure no zero in log scale
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
    plt.plot([min_size, max_size], [min_size, max_size], color="red", linestyle="--", label="Perfect Match (y = x)")
    plt.xlabel(f"GitHub {metric.value}")
    plt.ylabel(f"Gitea {metric.value}")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(min_size, max_size)
    plt.ylim(min_size, max_size)
    plt.title(f"Matched Repository Sizes: GitHub vs Gitea (Max Difference ≤ {max_difference})")
    plt.legend()

    plt.show()
