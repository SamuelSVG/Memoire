import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_language_distribution(df_github, df_gitea):
    # Remove missing values
    df_github = df_github.dropna(subset=['Language'])
    df_gitea = df_gitea.dropna(subset=['Language'])

    # Count repositories per language
    github_counts = df_github['Language'].value_counts()
    gitea_counts = df_gitea['Language'].value_counts()

    # Calculate the percentage of repositories for each language
    github_percentages = (github_counts / github_counts.sum()) * 100
    gitea_percentages = (gitea_counts / gitea_counts.sum()) * 100

    # Identify top N languages across both platforms
    top_languages = set(github_percentages.head(10).index) | set(gitea_percentages.head(10).index)

    # Filter data to include only the top languages
    github_percentages = github_percentages[github_percentages.index.isin(top_languages)]
    gitea_percentages = gitea_percentages[gitea_percentages.index.isin(top_languages)]

    # Merge the data to ensure all top languages appear
    top_languages_df = pd.concat([github_percentages, gitea_percentages], axis=1, keys=['GitHub', 'Gitea']).fillna(0)

    # Reset index and rename columns
    top_languages_df = top_languages_df.reset_index()
    top_languages_df.columns = ['Language', 'GitHub', 'Gitea']

    # Melt the data for plotting
    top_languages_df = top_languages_df.melt(id_vars='Language', var_name='Platform', value_name='Percentage')

    # Plot the bar chart
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Language', y='Percentage', hue='Platform', data=top_languages_df, palette='tab10')
    plt.title("Language Distribution on GitHub and Gitea")
    plt.xlabel("Programming Language")
    plt.ylabel("Percentage of Repositories (%)")
    plt.xticks(rotation=45)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.show()

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

def bin_data(df, platform, metric):
    metric_name = str(metric.name.lower())
    metric_column = metric.value

    df = df.dropna(subset=[metric_column]).copy()
    df.loc[:, metric_column] = pd.to_numeric(df[metric_column], errors='coerce')  # Convert to numeric

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

    return pd.DataFrame({'Platform': platform, f'{metric_name}_range': labels, 'Percentage': metric_percentages.values})

def plot_distribution(df_github, df_gitea, metric):
    # Process each platform
    # Get the appropriate function based on the metric
    metric_name = str(metric.name.lower())

    df_github_bins = bin_data(df_github, "GitHub", metric)
    df_gitea_bins = bin_data(df_gitea, "Gitea", metric)

    # Combine results
    df_bins = pd.concat([df_github_bins, df_gitea_bins])

    # Create the figure
    plt.figure(figsize=(12, 6))
    sns.barplot(x=f"{metric_name}_range", y="Percentage", hue="Platform", data=df_bins, palette="tab10")

    plt.xlabel(f"{metric_name} ranges")
    plt.ylabel("Percentage of Repositories (%)")
    plt.title(f"Repository {metric_name} Distribution Across Platforms")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(title="Platform")

    plt.show()
