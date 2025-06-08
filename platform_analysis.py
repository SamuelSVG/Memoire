from datetime import datetime
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.stats import mannwhitneyu
from psmpy import PsmPy
from metrics import Metrics
from psmpy.functions import cohenD

# Define bins and labels for different metrics
metric_bins = {
    "commit": [1, 10, 10**2, 10**3, 10**4, 10**5, np.inf],
    "size": [1, 10**5, 10**6, 10**7, 10**8, 10**9, np.inf],
    "branch": [0, 1, 5, 20, 100, np.inf],
    "contributor": [0, 1, 5, 10, 20, np.inf],
    "issue": [-np.inf, 0, 5, 10, 100, np.inf],
    "pull_request": [-np.inf, 0, 5, 10, 100, np.inf],
    "star": [-np.inf, 0, 5, 10, 100, np.inf],
    "created": [0, 14, 30, 90, 180, 360, np.inf],
}

metric_labels = {
    "commit": ["1-10", "11-100", "101-1K", "1K-10K", "10K-100K", ">100K"],
    "size": ["1B-100KB","100KB-1MB", "1MB-10MB", "10MB-100MB", "100MB-1GB", ">1GB"],
    "branch": ["1", "2-5", "6-20", "21-100", ">100"],
    "contributor": ["1", "2-5", "6-10", "11-20", ">20"],
    "issue": ["0", "1-5", "6-10", "11-100", ">100"],
    "pull_request": ["0", "1-5", "6-10", "11-100", ">100"],
    "star": ["0", "1-5", "6-10", "11-100", ">100"],
    "created": ["0-14 jours", "15-30 jours", "31-90 jours", "91-180 jours", "181-360 jours", ">360 jours"],
}

matching_features = {
    "git": ["#commits", "#branches", "#contributors", "size"],
    "platform": ["#stars", "#forks", "#issues", "#pull_requests"],
}

def get_clone_error_number(df_github, df_gitlab, df_gitea, df_forgejo):
    print(f"Github repositories that couldn't be cloned: {df_github['size'].isnull().sum()}")
    print(f"Gitlab repositories that couldn't be cloned: {df_gitlab['size'].isnull().sum()}")
    print(f"Gitea repositories that couldn't be cloned: {df_gitea['size'].isnull().sum()}")
    print(f"Forgejo repositories that couldn't be cloned: {df_forgejo['size'].isnull().sum()}")

def get_most_present_owner(df_github, df_gitlab, df_gitea, df_forgejo):
    print(f"Most present owner in GitHub: {df_github['owner'].value_counts().idxmax()} has {df_github['owner'].value_counts().max()} repositories")
    print(f"Most present owner in Gitlab: {df_gitlab['owner'].value_counts().idxmax()} has {df_gitlab['owner'].value_counts().max()} repositories")
    print(f"Most present owner in Gitea: {df_gitea['owner'].value_counts().idxmax()} has {df_gitea['owner'].value_counts().max()} repositories")
    print(f"Most present owner in Forgejo: {df_forgejo['owner'].value_counts().idxmax()} has {df_forgejo['owner'].value_counts().max()} repositories")

def get_unique_owner_number(df_github, df_gitlab, df_gitea, df_forgejo):
    print(f"Unique owners in GitHub: {df_github['owner'].nunique()}")
    print(f"Unique owners in Gitlab: {df_gitlab['owner'].nunique()}")
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
        n = len(lang_dict)
        threshold = 100 / (n + 1)
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
    df[metric_column] = df[metric_column].fillna("Pas de donnée")

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


def plot_alphanumeric_distribution(metric, df_github, df_gitlab, df_gitea, df_forgejo, n_boot=1000):
    """
    Plots the distribution of an alphanumeric metric (e.g., languages, licenses)
    across platforms using bootstrapped confidence intervals.
    """
    metric_name = metric.value

    # Generate bootstrapped samples
    df_github_bins = bin_alphanumeric_data(df_github, "GitHub", metric, n_boot)
    df_gitlab_bins = bin_alphanumeric_data(df_gitlab, "GitLab", metric, n_boot)
    df_gitea_bins = bin_alphanumeric_data(df_gitea, "Gitea", metric, n_boot)
    df_forgejo_bins = bin_alphanumeric_data(df_forgejo, "Forgejo", metric, n_boot)

    # Combine data
    df_bins = pd.concat([df_github_bins, df_gitlab_bins, df_gitea_bins, df_forgejo_bins])

    # Standardize platform order
    platform_order = ["GitHub", "GitLab", "Gitea", "Forgejo"]
    df_bins["platform"] = pd.Categorical(df_bins["platform"], categories=platform_order, ordered=True)

    # Identify top 10 categories across all platforms
    top_10_metric = (
        df_bins.groupby(metric_name, observed=True)["percentage"]
        .mean()
        .nlargest(10)
        .index
    )

    # Filter to top categories
    df_bins = df_bins[df_bins[metric_name].isin(top_10_metric)]

    # Group and compute bootstrapped confidence intervals
    summary_df = (
        df_bins
        .groupby([metric_name, "platform"], observed=True)["percentage"]
        .agg([
            ("mean", "mean"),
            ("ci_lower", lambda x: np.percentile(x, 2.5)),
            ("ci_upper", lambda x: np.percentile(x, 97.5))
        ])
        .reset_index()
    )

    # Sort metric categories for consistent plotting
    summary_df[metric_name] = pd.Categorical(summary_df[metric_name], categories=top_10_metric, ordered=True)

    # Plot
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(
        data=summary_df,
        x=metric_name,
        y="mean",
        hue="platform",
        palette="tab10",
        errorbar=None
    )

    # Add confidence interval error bars
    for i, row in summary_df.iterrows():
        x_val = list(summary_df[metric_name].cat.categories).index(row[metric_name])
        platform_offset = platform_order.index(row["platform"]) * 0.2 - 0.3  # adjust for number of platforms
        ax.errorbar(
            x=x_val + platform_offset,
            y=row["mean"],
            yerr=[[row["mean"] - row["ci_lower"]], [row["ci_upper"] - row["mean"]]],
            fmt='none',
            c='black',
            capsize=3
        )

    #plt.title(f"{metric_name} Distribution on the Main Platforms (with 95% Bootstrapped CI)")
    plt.xlabel("licence")
    plt.ylabel("Pourcentage de dépôts (%)")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(title="Platforme")
    plt.tight_layout()
    plt.savefig(f"Figures/{metric_name}_distribution.png", dpi=300, bbox_inches='tight')
    plt.show()

    df_github_clean = df_github.copy()
    df_gitlab_clean = df_gitlab.copy()
    df_gitea_clean = df_gitea.copy()
    df_forgejo_clean = df_forgejo.copy()

    if metric == Metrics.MAIN_LANGUAGE:
        df_github_clean["num_languages"] = df_github_clean["language_distribution"].apply(count_languages)
        df_gitlab_clean["num_languages"] = df_gitlab_clean["language_distribution"].apply(count_languages)
        df_gitea_clean["num_languages"] = df_gitea_clean["language_distribution"].apply(count_languages)
        df_forgejo_clean["num_languages"] = df_forgejo_clean["language_distribution"].apply(count_languages)

        # Print max, mean, and median number of languages for each DataFrame
        print(f"GitHub - Max number of languages: {df_github_clean['num_languages'].max()}, Mean number of languages: {df_github_clean['num_languages'].mean():.2f}, Median number of languages: {df_github_clean['num_languages'].median()}")
        print(f"GitLab - Max number of languages: {df_gitlab_clean['num_languages'].max()}, Mean number of languages: {df_gitlab_clean['num_languages'].mean():.2f}, Median number of languages: {df_gitlab_clean['num_languages'].median()}")
        print(f"Gitea - Max number of languages: {df_gitea_clean['num_languages'].max()}, Mean number of languages: {df_gitea_clean['num_languages'].mean():.2f}, Median number of languages: {df_gitea_clean['num_languages'].median()}")
        print(f"Forgejo - Max number of languages: {df_forgejo_clean['num_languages'].max()}, Mean number of languages: {df_forgejo_clean['num_languages'].mean():.2f}, Median number of languages: {df_forgejo_clean['num_languages'].median()}")

    if metric == Metrics.LICENSE:
        github_counts = df_github[metric.value].value_counts().get('No metric', 0)
        gitlab_counts = df_gitlab[metric.value].value_counts().get('No metric', 0)
        gitea_counts = df_gitea[metric.value].value_counts().get('No metric', 0)
        forgejo_counts = df_forgejo[metric.value].value_counts().get('No metric', 0)
        print(f"GitHub repositories that have a license: {((len(df_github) - github_counts) / len(df_github)) * 100:.2f}%")
        print(f"Gitlab repositories that have a license: {((len(df_gitlab) - gitlab_counts) / len(df_gitlab)) * 100:.2f}%")
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
    if metric == Metrics.CREATED:
        df[metric_column] = pd.to_datetime(df[metric_column], errors='coerce').dt.tz_localize(None)
        reference_date = pd.to_datetime(df["updated"], errors='coerce').dt.tz_localize(None)
        df[metric_column] = (reference_date - df[metric_column]).dt.days + 1

    df.loc[:, metric_column] = pd.to_numeric(df[metric_column], errors='coerce')

    bins = metric_bins.get(metric_name)
    labels = metric_labels.get(metric_name)
    bin_col = f"{metric_name}_range"

    df[bin_col] = pd.cut(df[metric_column], bins=bins, labels=labels, include_lowest=True)

    # Ensure bin column is a categorical with correct order
    df[bin_col] = pd.Categorical(df[bin_col], categories=labels, ordered=True)

    # Bootstrapping
    bootstrapped_data = []
    for _ in range(n_boot):
        sample = df.sample(frac=1, replace=True)
        bin_counts = sample[bin_col].value_counts().reindex(labels, fill_value=0)
        total_repos = bin_counts.sum()
        bin_percentages = (bin_counts / total_repos) * 100

        for bin_label, percentage in bin_percentages.items():
            bootstrapped_data.append({
                "platform": platform,
                bin_col: bin_label,
                "percentage": percentage
            })

    result_df = pd.DataFrame(bootstrapped_data)

    # Set platform as ordered categorical
    platform_order = ["GitHub", "GitLab", "Gitea", "Forgejo"]
    result_df["platform"] = pd.Categorical(result_df["platform"], categories=platform_order, ordered=True)

    # Set bin column as ordered categorical again (in case lost during DataFrame creation)
    result_df[bin_col] = pd.Categorical(result_df[bin_col], categories=labels, ordered=True)

    return result_df


def plot_numeric_distribution(df_github, df_gitlab, df_gitea, df_forgejo, metric, n_boot=1000):
    """
    Plot the distribution of a given metric across different platforms with bootstrapped 95% confidence intervals.
    """
    metric_name = str(metric.name.lower())
    bin_col = f"{metric_name}_range"

    # Get bootstrapped data
    df_github_bins = bin_data(df_github, "GitHub", metric, n_boot)
    df_gitlab_bins = bin_data(df_gitlab, "GitLab", metric, n_boot)
    df_gitea_bins = bin_data(df_gitea, "Gitea", metric, n_boot)
    df_forgejo_bins = bin_data(df_forgejo, "Forgejo", metric, n_boot)

    df_bins = pd.concat([df_github_bins, df_gitlab_bins, df_gitea_bins, df_forgejo_bins])

    # Compute 95% CI summary
    summary_df = (
        df_bins
        .groupby([bin_col, "platform"], observed=True)["percentage"]
        .agg([
            ("mean", "mean"),
            ("ci_lower", lambda x: np.percentile(x, 2.5)),
            ("ci_upper", lambda x: np.percentile(x, 97.5))
        ])
        .reset_index()
    )

    # Ensure categorical order is preserved in the summary
    platform_order = ["GitHub", "GitLab", "Gitea", "Forgejo"]
    summary_df["platform"] = pd.Categorical(summary_df["platform"], categories=platform_order, ordered=True)
    summary_df[bin_col] = pd.Categorical(summary_df[bin_col], categories=metric_labels[metric_name], ordered=True)

    # Plot
    plt.figure(figsize=(12, 6))
    width = 0.2
    bin_positions = np.arange(len(metric_labels[metric_name]))

    for idx, platform in enumerate(platform_order):
        platform_data = summary_df[summary_df["platform"] == platform]
        positions = bin_positions + (idx - 1.5) * width
        plt.bar(
            positions,
            platform_data["mean"],
            width=width,
            label=platform,
            yerr=[
                platform_data["mean"] - platform_data["ci_lower"],
                platform_data["ci_upper"] - platform_data["mean"]
            ],
            capsize=4
        )

    plt.xticks(bin_positions, metric_labels[metric_name])
    plt.xlabel(f"Nombre de pull requests des dépôts")
    plt.ylabel("Pourcentage de dépôts (%)")
    #plt.title(f"Repository {metric_name} Distribution Across Platforms (with 95% Bootstrapped CI)")
    plt.legend(title="Platforme")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"Figures/{metric_name}_distribution.png", dpi=300, bbox_inches='tight')
    plt.show()

    if metric == Metrics.CREATED:
        # Convert date columns to Unix timestamps
        df_github[metric.value] = (pd.to_datetime(df_github["updated"], errors='coerce').astype(int) / 10 ** 9 -
                                   pd.to_datetime(df_github["created"], errors='coerce').astype(int) / 10 ** 9) / (60 * 60 * 24)
        df_gitlab[metric.value] = (pd.to_datetime(df_gitlab["updated"], errors='coerce').astype(int) / 10 ** 9 -
                                   pd.to_datetime(df_gitlab["created"], errors='coerce').astype(int) / 10 ** 9) / (60 * 60 * 24)
        df_gitea[metric.value] = (pd.to_datetime(df_gitea["updated"], errors='coerce').astype(int) / 10 ** 9 -
                                   pd.to_datetime(df_gitea["created"], errors='coerce').astype(int) / 10 ** 9) / (60 * 60 * 24)
        df_forgejo[metric.value] = (pd.to_datetime(df_forgejo["updated"], errors='coerce').astype(int) / 10 ** 9 -
                                   pd.to_datetime(df_forgejo["created"], errors='coerce').astype(int) / 10 ** 9) / (60 * 60 * 24)

    # Print max, mean, and median values for each DataFrame
    print(f"GitHub - Max {metric_name}: {int(df_github[metric.value].max())}, Mean {metric_name}: {int(df_github[metric.value].mean())}, Median {metric_name}: {int(df_github[metric.value].median())}")
    print(f"GitLab - Max {metric_name}: {int(df_gitlab[metric.value].max())}, Mean {metric_name}: {int(df_gitlab[metric.value].mean())}, Median {metric_name}: {int(df_gitlab[metric.value].median())}")
    print(f"Gitea - Max {metric_name}: {int(df_gitea[metric.value].max())}, Mean {metric_name}: {int(df_gitea[metric.value].mean())}, Median {metric_name}: {int(df_gitea[metric.value].median())}")
    print(f"Forgejo - Max {metric_name}: {int(df_forgejo[metric.value].max())}, Mean {metric_name}: {int(df_forgejo[metric.value].mean())}, Median {metric_name}: {int(df_forgejo[metric.value].median())}")

    # Perform Mann-Whitney U tests
    platform_data = {
        "GitHub": df_github[metric.value].dropna(),
        "GitLab": df_gitlab[metric.value].dropna(),
        "Gitea": df_gitea[metric.value].dropna(),
        "Forgejo": df_forgejo[metric.value].dropna()
    }

    print(f"\nMann-Whitney U test results for '{metric_name}':")
    for (name1, data1), (name2, data2) in itertools.combinations(platform_data.items(), 2):
        stat, p = mannwhitneyu(data1, data2, alternative='two-sided')
        print(f"{name1} vs {name2}: U={stat:.2f}, p-value={p:.4f}")

def plot_step_lines(df_github, df_gitlab, df_gitea, df_forgejo, metric):
    plt.figure(figsize=(12, 6))

    # Dictionary to store dataframes and their platform labels
    platforms = {
        "GitHub": df_github,
        "GitLab": df_gitlab,
        "Gitea": df_gitea,
        "Forgejo": df_forgejo
    }

    # Create a new DataFrame to store all data
    combined_data = []

    for label, df in platforms.items():
        # Convert to datetime and extract date
        df[metric.value] = pd.to_datetime(df[metric.value])
        df["date"] = df[metric.value].dt.date

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
    plt.xlabel("Date de mise à jour")
    plt.ylabel("Nombre cumulé de dépôts mis à jour")
    #plt.title("Cumulative Repositories Created Over Time")
    plt.xticks(rotation=45)
    plt.legend(title="Platforme")
    plt.grid()

    plt.savefig("Figures/cumulative_repositories_updated_15.png", dpi=300, bbox_inches='tight')
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
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", center=0)
    #plt.title("Correlation Matrix of Repository Metrics")
    plt.savefig("Figures/correlation_matrix_forgejo.png", dpi=300, bbox_inches='tight')
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

def propensity_score_matching(
        df_platform1,
        df_platform2,
        treated_platform,
        outcome_col,
        covariates,
        caliper=0.1,
        verbose=True,
        plot_balance=True
):
    assert covariates is not None, "You must specify covariates for matching."

    df_platform1 = df_platform1.copy()
    df_platform2 = df_platform2.copy()

    df_platform1 = df_platform1.dropna(subset=covariates + ["platform", outcome_col])
    df_platform2 = df_platform2.dropna(subset=covariates + ["platform", outcome_col])

    if outcome_col == Metrics.CREATED.value or Metrics.CREATED.value in covariates:
        df_platform1[Metrics.CREATED.value] = pd.to_datetime(df_platform1[Metrics.CREATED.value], errors='coerce').dt.tz_localize(None)
        df_platform2[Metrics.CREATED.value] = pd.to_datetime(df_platform2[Metrics.CREATED.value], errors='coerce').dt.tz_localize(None)
        reference_date = datetime.strptime("09/05/2025", "%d/%m/%Y")
        df_platform1[Metrics.CREATED.value] = (reference_date - df_platform1[Metrics.CREATED.value]).dt.days
        df_platform2[Metrics.CREATED.value] = (reference_date - df_platform2[Metrics.CREATED.value]).dt.days

    df = pd.concat([df_platform1, df_platform2], ignore_index=True)
    # Create binary treatment column
    df['treatment'] = (df["platform"] == treated_platform).astype(int)
    id_col = 'id'

    # Only keep relevant columns
    columns_to_keep = [id_col, 'treatment', outcome_col] + covariates
    df = df[columns_to_keep]

    # Initialize PsmPy
    psm = PsmPy(df, treatment='treatment', indx=id_col, exclude=[outcome_col])

    # Estimate propensity scores
    psm.logistic_ps(balance=False)

    # Perform matching
    psm.knn_matched(matcher='propensity_logit', replacement=False, caliper=caliper, drop_unmatched=True)

    # Get matched data
    matched_df = psm.df_matched

    # Merge outcome column back
    matched_df = matched_df.merge(df[[id_col, outcome_col]], on=id_col, how='left')

    # Run Mann-Whitney U test
    treated = matched_df[matched_df['treatment'] == 1][outcome_col]
    control = matched_df[matched_df['treatment'] == 0][outcome_col]
    u_stat, p_value = mannwhitneyu(treated, control, alternative='two-sided')

    if verbose:
        print("\nMatched Sample Sizes:", "\nTreated:", len(treated), " | Control:", len(control))
        print(f"{outcome_col} Median (Treated): {treated.median():.2f}")
        print(f"{outcome_col} Median (Control): {control.median():.2f}")
        print(f"Mann-Whitney U Test p-value: {p_value:.4f}")
        print("\nCovariate Balance After Matching (Cohen's D):")
        for cov in covariates:
            d = cohenD(matched_df, 'treatment', cov)
            print(f"  {cov}: {d:.4f}")

    # Plot covariate balance
    if plot_balance:
        psm.effect_size_plot(title="", save=True)
