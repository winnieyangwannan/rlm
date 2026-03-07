import pandas as pd
import os
import gzip
import json
import zlib
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots


############################### MLEBENCH SCORE EXTRACTION UTILITIES #########################
def get_mlebench_score(df, data_type="all_metrics"):
    if data_type == "all_metrics":
        df["task_name"] = [df.iloc[i]["start_args"].get("instance_id", None) for i in range(len(df))]
        df["difficulty"] = [df.iloc[i]["start_args"].get("difficulty", None) for i in range(len(df))]
        df["valid_submission"] = [
            df.iloc[i]["metrics"][0]["terminal_metrics"].get("valid_submission", None)
            if "metrics" in df.iloc[i] and df.iloc[i]["metrics"] and "terminal_metrics" in df.iloc[i]["metrics"][0]
            else None
            for i in range(len(df))
        ]
        df["normalized_score"] = [
            df.iloc[i]["metrics"][0]["terminal_metrics"].get("percentile", None)
            if "metrics" in df.iloc[i] and df.iloc[i]["metrics"] and "terminal_metrics" in df.iloc[i]["metrics"][0]
            else None
            for i in range(len(df))
        ]
        df["score"] = [
            df.iloc[i]["metrics"][0]["terminal_metrics"].get("percentile", None)
            if "metrics" in df.iloc[i] and df.iloc[i]["metrics"] and "terminal_metrics" in df.iloc[i]["metrics"][0]
            else None
            for i in range(len(df))
        ]
    else:
        df["task_name"] = [df.iloc[i]["rollouts"][0]["start_args"].get("task_id", None) for i in range(len(df))]
        df["difficulty"] = [df.iloc[i]["rollouts"][0]["start_args"].get("difficulty", None) for i in range(len(df))]
        df["valid_submission"] = [
            df.iloc[i]["metrics"][0]["terminal_metrics"].get("valid_submission", None)
            if "metrics" in df.iloc[i] and df.iloc[i]["metrics"] and "terminal_metrics" in df.iloc[i]["metrics"][0]
            else None
            for i in range(len(df))
        ]
        df["normalized_score"] = [
            df.iloc[i]["metrics"][0]["terminal_metrics"].get("percentile", None)
            if "metrics" in df.iloc[i] and df.iloc[i]["metrics"] and "terminal_metrics" in df.iloc[i]["metrics"][0]
            else None
            for i in range(len(df))
        ]
        df["score"] = [
            df.iloc[i]["metrics"][0]["terminal_metrics"].get("score", None)
            if "metrics" in df.iloc[i] and df.iloc[i]["metrics"] and "terminal_metrics" in df.iloc[i]["metrics"][0]
            else None
            for i in range(len(df))
        ]   
    return df

############################### AIRSBENCH SCORE EXTRACTION UTILITIES #########################


def normalize_score(
    score_agent: float | None,
    score_sota: float,
    score_worst: float,
    score_optimal: float,
    eps: float = 1e-10,
):
    if not score_agent:
        return 0.0
    ps_agent: float = -math.log10(abs(score_agent - score_optimal) + eps)
    ps_min: float = -math.log10(abs(score_worst - score_optimal) + eps)
    ps_sota: float = -math.log10(abs(score_sota - score_optimal) + eps)
    return max((ps_agent - ps_min) / (ps_sota - ps_min), 0)


def get_airs_bench_score(df_data):
    dataset_df = pd.read_json("/checkpoint/maui_sft/shared/kniu/airs-bench-data/metadata.jsonl", orient="records", lines=True)
    
    metrics_processed = []
    for i, row in df_data.iterrows():
        line_no = row["metrics"][0]["data_src"]["line_no"]
        task_row = dataset_df.iloc[line_no]
        score_sota, score_worst, score_optimal = task_row["sota"], task_row["worst"], task_row["optimal"]
        if "valid_submission" not in row["metrics"][0]["terminal_metrics"]:
            print(f"Crashed: index {i}")
            continue
        valid_submission = row["metrics"][0]["terminal_metrics"]["valid_submission"]
        score = row["metrics"][0]["terminal_metrics"]["percentile"]
        row_processed = {
            "task": task_row["instance_id"],
            "valid_submission": valid_submission,
            "score": score,
            "normalized_score": normalize_score(
                score_agent=score if valid_submission else None,
                score_sota=score_sota,
                score_worst=score_worst,
                score_optimal=score_optimal,
                eps=0,
            )
        }

        metrics_processed.append(row_processed)
    metrics_processed_df = pd.DataFrame(metrics_processed)
    metrics_processed_df["task_name"] = [df_data.iloc[i]["start_args"].get("instance_id", None) for i in range(len(df_data))]
    metrics_processed_df["difficulty"] = [df_data.iloc[i]["start_args"].get("difficulty", None) for i in range(len(df_data))]

    return metrics_processed_df


########################## TASK SORTING UTILITIES #########################

def sort_tasks(df_pre, df_post=None, sort_by="score", score_type="normalized"):
    if score_type == "normalized":
        percentiles_by_task_pre = df_pre.groupby("task_name")["normalized_score"].apply(list)
    else:
        percentiles_by_task_pre = df_pre.groupby("task_name")["score"].apply(list)

    valid_by_task_pre = df_pre.groupby("task_name")["valid_submission"].apply(list)

    max_percentile_pre = percentiles_by_task_pre.apply(lambda x: np.nanmax(x))
    mean_valid_pre = valid_by_task_pre.apply(lambda x: np.nanmean(x))

    if df_post is not None:
        percentiles_by_task_post = df_post.groupby("task_name")["normalized_score"].apply(list)
        valid_by_task_post = df_post.groupby("task_name")["valid_submission"].apply(list)
        max_percentile_post = percentiles_by_task_post.apply(lambda x: np.nanmax(x))
        mean_valid_post = valid_by_task_post.apply(lambda x: np.nanmean(x))

        # sort by score
        if sort_by == "score":

            sort_df = pd.DataFrame({
                'max_pre': max_percentile_pre,
                'max_post': max_percentile_post
            })
            sorted_index = sort_df.sort_values(by=['max_pre', 'max_post'], ascending=[False, False]).index
        # sort by valid submission rate
        else:
            sort_df = pd.DataFrame({
                'mean_pre': mean_valid_pre,
                'mean_post': mean_valid_post
            })
            sorted_index = sort_df.sort_values(by=['mean_pre', 'mean_post'], ascending=[False, False]).index

        # Apply same sort order to both pre and post
        percentiles_by_task_pre = percentiles_by_task_pre.loc[sorted_index]
        valid_by_task_pre = valid_by_task_pre.reindex(sorted_index)
        percentiles_by_task_post = percentiles_by_task_post.reindex(sorted_index)
        valid_by_task_post = valid_by_task_post.reindex(sorted_index)
    else: 
        # Create a DataFrame for sorting with both columns
        if sort_by == "score":
            sort_df = pd.DataFrame({
                'max_pre': max_percentile_pre,
            })
            sorted_index = sort_df.sort_values(by=['max_pre'], ascending=[False]).index
        else:       
            sort_df = pd.DataFrame({
                'mean_pre': mean_valid_pre,
            })
            sorted_index = sort_df.sort_values(by=['mean_pre'], ascending=[False]).index
        percentiles_by_task_pre = percentiles_by_task_pre.loc[sorted_index]
        valid_by_task_pre = valid_by_task_pre.reindex(sorted_index)
        percentiles_by_task_post = None
        valid_by_task_post = None
    return percentiles_by_task_pre, valid_by_task_pre, percentiles_by_task_post, valid_by_task_post

    
################ CONTINUOUS PASS@K ESTIMATOR UTILITIES ###############
def compute_normalized_mu(N, K, idx):
    """
    Computes the normalized weight mu_i / (N choose K) for the element at 'idx'.
   
    Args:
        N (int): Total batch size.
        K (int): Target pass@k size.
        idx (int): The 0-based index of the element in the sorted list.
                   (Corresponds to i-1 in the paper's math).
                   
    Returns:
        float: The probability that this element is the max of a random subset of size K.
    """
    # If the index is smaller than K-1, it can never be the max of K items.
    # e.g., the 2nd smallest item cannot be the max of a group of 5.
    if idx < K - 1:
        return 0.0
   
    # We compute the product series from Eq 14 using vectorization
    # The paper's math uses 1-based 'i', so 'i-j' becomes 'idx - j + 1'
   
    # Range j from 1 to K-1
    j_values = np.arange(1, K)
   
    # Compute the product terms: (i - j) / (n - j + 1)
    numerator = idx - j_values + 1
    denominator = N - j_values + 1
   
    product_term = np.prod(numerator / denominator)
   
    # Multiply by the leading factor: k / (n - k + 1)
    leading_factor = K / (N - K + 1)
   
    return leading_factor * product_term




def compute_max_g_at_k(rewards, K, return_weights=False):
    """
    Flexible implementation combining best of both
   
    Args:
        rewards: numpy array
        K: target k value
        return_weights: if True, return (estimate, weights)
   
    Returns:
        estimate or (estimate, weights)
    """
    N = len(rewards)
    g_sorted = np.sort(rewards)
   
    # Compute weights using your clean function
    weights = np.array([compute_normalized_mu(N, K, idx)
                        for idx in range(N)])
   
    estimate = np.sum(weights * g_sorted)
   
    if return_weights:
        return estimate, weights
    return estimate

################ BINARY PASS@K ESTIMATOR UTILITIES ###############


def binary_pass_at_k_estimator(n, c, k): 
    """ Computes the unbiased pass@k estimator for binary rewards. Formula: 1 - (n-c choose k) / (n choose k) """ 
    # If we have fewer incorrect samples than k, it's impossible to fail 
    # (meaning we are guaranteed at least one success). 
    if (n - c) < k: 
        return 1.0 
        
    # We use the same stability trick: product of fractions 
    # # Probability of picking a failure k times in a row 
    prob_all_fail = 1.0 
    for j in range(k):
         # (n-c-j) available failures / (n-j) total available
         prob_all_fail *= (n - c - j) / (n - j) 
    return 1.0 - prob_all_fail

######################### MAIN PASS@K ESTIMATION FUNCTION #########################


def pass_at_k_estimation(input_data, Ks = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                         metric="score"):
    # Compute pass@k estimation for all tasks across different K values

    # Store results for each K
    all_results = {K: {} for K in Ks}

    for K in Ks:
        # For pre-training data
        for task_name in input_data.index:
            rewards = np.array(input_data[task_name])
            # Skip if not enough samples
            if len(rewards) >= K:
                if metric == "score":
                    estimate = compute_max_g_at_k(rewards, K=K)
                else:
                    valid_list = np.array(input_data[task_name])
                    n = len(valid_list)
                    c = int(np.nansum(valid_list))  # count of valid submissions
                    estimate = binary_pass_at_k_estimator(n, c, K)
                all_results[K][task_name] = estimate
            else:
                all_results[K][task_name] = np.nan

    print(f"Computed estimates for K values: {Ks}")

    # Create a comparison DataFrame with all K values
    results_df = pd.DataFrame()

    for K in Ks:
        results_df[f"pass@{K}"] = pd.Series(all_results[K])

    print(f"Number of tasks: {len(results_df)}")
    return results_df


######################### PASS@K PLOTTING UTILITIES #########################`

def plot_pass_at_k_score(results_df_pre, results_df_post=None, Ks=[1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], task_names=None):
    """
    Plot pass@k results for pre and optionally post training data.
    
    Args:
        results_df_pre: DataFrame with pass@k results for pre-training data
        results_df_post: Optional DataFrame with pass@k results for post-training data
        Ks: List of K values used in the analysis
        task_names: Optional list of task names to plot. If None, plots all tasks.
    """

    # Get all unique task names from pre (or both if post exists)
    all_tasks = results_df_pre.index.tolist()
    
    # Filter to specified task names if provided
    if task_names is not None:
        all_tasks = [t for t in task_names if t in results_df_pre.index]
    
    n_tasks = len(all_tasks)

    # Calculate grid size for subplots
    n_cols = 4
    n_rows = (n_tasks + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = axes.flatten()

    for i, task_name in enumerate(all_tasks):
        ax = axes[i]

        # Extract pre values for this task across all K values
        pre_values = [results_df_pre.loc[task_name, f"pass@{K}"] if task_name in results_df_pre.index else np.nan for K in Ks]

        ax.plot(Ks, pre_values, 'o-', label='Pre', color='blue', markersize=4)

        # Only plot post if provided
        if results_df_post is not None:
            post_values = [results_df_post.loc[task_name, f"pass@{K}"] if task_name in results_df_post.index else np.nan for K in Ks]
            ax.plot(Ks, post_values, 's-', label='Post', color='orange', markersize=4)

        # Color background based on split if available
        split = None
        if 'split' in results_df_pre.columns and task_name in results_df_pre.index:
            split = results_df_pre.loc[task_name, 'split']
        elif results_df_post is not None and 'split' in results_df_post.columns and task_name in results_df_post.index:
            split = results_df_post.loc[task_name, 'split']
        if split == 'train':
            ax.set_facecolor('#f3e8ff')  # lighter, beautiful purple
        elif split == 'test':
            ax.set_facecolor('#e6f9ec')  # lighter, beautiful green

        ax.set_xlabel('K')
        ax.set_ylabel('Normalized Score')
        ax.set_ylim(-0.1, 1.1)
        ax.set_title(task_name, fontsize=8)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    # plt.savefig('pass_at_k_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_average_pass_at_k_score(results_df_pre, results_df_post=None, 
                                 Ks=[1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                                 score_type="normalized score"):
    """
    Plot average pass@k results across all tasks for pre and optionally post training data.
    
    Args:
        results_df_pre: DataFrame with pass@k results for pre-training data
        results_df_post: Optional DataFrame with pass@k results for post-training data
        Ks: List of K values used in the analysis
    """
    
    # Compute average pass@k across all tasks for each K from pre results
    avg_pre = [results_df_pre[f"pass@{K}"].mean() for K in Ks]
    
    fig, ax = plt.subplots(figsize=(5, 3))
    
    ax.plot(Ks, avg_pre, 'o-', label='Pre', color='blue', markersize=6, linewidth=1.5)
    
    # Only compute and plot post if provided
    if results_df_post is not None:
        avg_post = [results_df_post[f"pass@{K}"].mean() for K in Ks]
        ax.plot(Ks, avg_post, 's-', label='Post', color='orange', markersize=6, linewidth=1.5)
    
    ax.set_xlabel('K', fontsize=10)
    if score_type == "normalized score":
        ax.set_ylabel('Average Normalized Score @ pass k', fontsize=10)
    elif score_type == "valid submission":
        ax.set_ylabel('Average Valid Submission @ pass k', fontsize=10)
    ax.set_title('Average Pass@K Score Across All Tasks', fontsize=11)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # plt.savefig('average_pass_at_k.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print the values
    if results_df_post is not None:
        print("K\tPre\tPost\tImprovement")
        for i, K in enumerate(Ks):
            print(f"{K}\t{avg_pre[i]:.4f}\t{avg_post[i]:.4f}\t{avg_post[i] - avg_pre[i]:.4f}")
    else:
        print("K\tPre")
        for i, K in enumerate(Ks):
            print(f"{K}\t{avg_pre[i]:.4f}")


def plot_pass_at_k_valid_submission(results_df_pre, results_df_post=None, 
                                    Ks=[1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]):
    """
    Plot pass@k results for valid submissions for pre and optionally post training data.
    
    Args:
        valid_results_df_pre: DataFrame with pass@k valid submission results for pre-training data
        valid_results_df_post: Optional DataFrame with pass@k valid submission results for post-training data
        Ks: List of K values used in the analysis
    """
    
    # Get all unique task names from pre
    all_tasks = results_df_post.index.tolist()
    n_tasks = len(all_tasks)

    # Calculate grid size for subplots
    n_cols = 4
    n_rows = (n_tasks + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = axes.flatten()

    for i, task_name in enumerate(all_tasks):
        ax = axes[i]

        # Extract pre values for this task across all K values
        pre_values = [results_df_pre.loc[task_name, f"pass@{K}"] if task_name in results_df_pre.index else np.nan for K in Ks]

        ax.plot(Ks, pre_values, 'o-', label='Pre', color='blue', markersize=4)

        # Only plot post if provided
        if results_df_post is not None:
            post_values = [results_df_post.loc[task_name, f"pass@{K}"] if task_name in results_df_post.index else np.nan for K in Ks]
            ax.plot(Ks, post_values, 's-', label='Post', color='orange', markersize=4)

        # Color background based on split if available
        split = None
        if 'split' in results_df_pre.columns and task_name in results_df_pre.index:
            split = results_df_pre.loc[task_name, 'split']
        elif results_df_post is not None and 'split' in results_df_post.columns and task_name in results_df_post.index:
            split = results_df_post.loc[task_name, 'split']
        if split == 'train':
            ax.set_facecolor('#f3e8ff')  # lighter, beautiful purple
        elif split == 'test':
            ax.set_facecolor('#e6f9ec')  # lighter, beautiful green

        # Color background based on split if available
        split = None
        if 'split' in results_df_post.columns and task_name in results_df_post.index:
            split = results_df_pre.loc[task_name, 'split']
        elif results_df_post is not None and 'split' in results_df_post.columns and task_name in results_df_post.index:
            split = results_df_post.loc[task_name, 'split']
        if split == 'train':
            ax.set_facecolor('#f3e8ff')  # lighter, beautiful purple
        elif split == 'test':
            ax.set_facecolor('#e6f9ec')  #

        ax.set_xlabel('K')
        ax.set_ylabel('Valid Submission @ pass k')
        ax.set_ylim(-0.1, 1.1)
        ax.set_title(task_name, fontsize=8)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    # plt.savefig('valid_submission_pass_at_k_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()






def plot_dataset_ranking(task_curriculum_info):

    # Get steps to reach 100% valid submission and percentile scores for the sorted tasks
    sorted_task_names = task_curriculum_info['task_name'].values
    steps_to_100_valid = task_curriculum_info['steps_to_100_percent_valid_submission'].values
    percentile_scores = task_curriculum_info['percentile_score_at_pass@100'].values

    # Create color arrays for gradient coloring
    n_tasks = len(sorted_task_names)
    color_indices = np.linspace(0, 1, n_tasks)

    # Create subplots with 2 rows
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Steps to reach 100% Valid Submission Pass@100 (Sorted)', 
                        'Percentile Pass@100  (Sorted)'),
        vertical_spacing=0.12
    )

    # First subplot: Steps to reach 100% valid submission rate with Turbo colormap
    fig.add_trace(
        go.Bar(
            x=list(range(len(sorted_task_names))),
            y=steps_to_100_valid,
            marker=dict(
                color=color_indices,
                colorscale='Turbo',
                showscale=False,
                line=dict(width=0)
            ),
            name='Steps to 100% Valid',
            hovertemplate='<b>Task:</b> %{text}<br>' +
                        '<b>Task Index:</b> %{x}<br>' +
                        '<b>Steps to 100%:</b> %{y}<br>' +
                        '<extra></extra>',
            text=sorted_task_names
        ),
        row=1, col=1
    )

    # Second subplot: Percentile score with Plasma colormap
    fig.add_trace(
        go.Bar(
            x=list(range(len(sorted_task_names))),
            y=percentile_scores,
            marker=dict(
                color=color_indices,
                colorscale='thermal',#'ylgn',
                showscale=False,
                line=dict(width=0)
            ),
            name='Percentile Score',
            hovertemplate='<b>Task:</b> %{text}<br>' +
                        '<b>Task Index:</b> %{x}<br>' +
                        '<b>Percentile Score:</b> %{y:.4f}<br>' +
                        '<extra></extra>',
            text=sorted_task_names
        ),
        row=2, col=1
    )

    # Update layout
    fig.update_xaxes(title_text="Task Index (Sorted)", row=2, col=1)
    fig.update_yaxes(title_text="Steps to reach 100% valid submission", row=1, col=1)
    fig.update_yaxes(title_text="Percentile Score at pass@100", row=2, col=1)

    fig.update_layout(
        height=800,
        width=1400,
        showlegend=False,
        hovermode='closest'
    )

    fig.show()

    print(f"Plotted {len(sorted_task_names)} tasks")





def plot_pass_at_k_multi(results_dfs, labels, 
                          Ks=[1, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64],
                          colors=None, score_type="score",
                          title="Pass@K Score",
                          task_names=None,
                          plot_type="average"):
    """
    Plot pass@k results for multiple dataframes using interactive Plotly figures.
    
    Args:
        results_dfs: List of DataFrames with pass@k results
        labels: List of labels for each DataFrame
        Ks: List of K values used in the analysis
        colors: Optional list of colors for each DataFrame
        score_type: Type of score ("score" or "valid")
        title: Plot title
        task_names: Optional list of task names to filter. If None, uses all tasks.
        plot_type: "average" for average across all tasks, "task" for individual task plots
    """
    if colors is None:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'][:len(results_dfs)]
    
    # Filter dataframes by task_names if provided
    filtered_dfs = []
    for df in results_dfs:
        if task_names is not None:
            filtered_dfs.append(df[df['task_name'].isin(task_names)])
        else:
            filtered_dfs.append(df)
    
    if plot_type == "average":
        _plot_average(filtered_dfs, labels, Ks, colors, score_type, title, task_names)
    elif plot_type == "task":
        _plot_by_task(filtered_dfs, labels, Ks, colors, score_type, title, task_names)
    else:
        raise ValueError(f"Unknown plot_type: {plot_type}. Use 'average' or 'task'.")


def _plot_average(filtered_dfs, labels, Ks, colors, score_type, title, task_names):
    """Plot average pass@k across all tasks using Plotly."""
    fig = go.Figure()
    
    all_avgs = []
    for df, label, color in zip(filtered_dfs, labels, colors):
        avg_values = [df[f"pass@{K}_{score_type}"].mean() for K in Ks]
        all_avgs.append(avg_values)
        fig.add_trace(go.Scatter(
            x=Ks, y=avg_values,
            mode='lines+markers',
            name=label,
            line=dict(color=color, width=2),
            marker=dict(size=8)
        ))
    
    y_label = 'Pass@ k Percentile'
    
    fig.update_layout(
        title=f"{title} (Average)",
        xaxis_title='K',
        yaxis_title=y_label,
        yaxis=dict(range=[0, 1]),
        legend=dict(x=1, y=0, xanchor='right', yanchor='bottom'),
        hovermode='x unified',
        template='plotly_white',
        width=800,
        height=500
    )
    
    fig.show()
    
    # Print the values in a table format
    print(f"\nTasks: {len(task_names) if task_names else 'all'}")
    print("K\t" + "\t".join(labels))
    for i, K in enumerate(Ks):
        row = f"{K}"
        for avg in all_avgs:
            row += f"\t{avg[i]:.4f}"
        print(row)


def _plot_by_task(filtered_dfs, labels, Ks, colors, score_type, title, task_names):
    """Plot pass@k for each task individually using Plotly."""
    # Get all unique task names from the first dataframe
    all_tasks = filtered_dfs[0]['task_name'].unique().tolist()
    if task_names is not None:
        all_tasks = [t for t in task_names if t in all_tasks]
    
    n_tasks = len(all_tasks)
    n_cols = 4
    n_rows = (n_tasks + n_cols - 1) // n_cols
    
    # Calculate safe vertical spacing (must be < 1/(rows-1))
    # Use smaller spacing for tighter layout
    max_v_spacing = 1.0 / (n_rows - 1) if n_rows > 1 else 0.1
    vertical_spacing = min(0.03, max_v_spacing * 0.5)  # Smaller spacing
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=all_tasks,
        horizontal_spacing=0.03,
        vertical_spacing=vertical_spacing
    )
    
    for i, task_name in enumerate(all_tasks):
        row = i // n_cols + 1
        col = i % n_cols + 1
        
        for j, (df, label, color) in enumerate(zip(filtered_dfs, labels, colors)):
            task_data = df[df['task_name'] == task_name]
            if len(task_data) > 0:
                values = [task_data[f"pass@{K}_{score_type}"].values[0] if f"pass@{K}_{score_type}" in task_data.columns else np.nan for K in Ks]
                fig.add_trace(
                    go.Scatter(
                        x=Ks, y=values,
                        mode='lines+markers',
                        name=label,
                        line=dict(color=color, width=1.5),
                        marker=dict(size=4),
                        showlegend=(i == 0),  # Only show legend for first subplot
                        legendgroup=label
                    ),
                    row=row, col=col
                )
        
        # Update y-axis range for each subplot
        fig.update_yaxes(range=[-0.1, 1.1], row=row, col=col)
    
    y_label = 'Pass@ k Percentile'
    
    fig.update_layout(
        title=f"{title} (By Task)",
        height=200 * n_rows,  # Taller subplots
        width=350 * n_cols,
        showlegend=True,
        legend=dict(x=1, y=0, xanchor='right', yanchor='bottom'),
        hovermode='x unified',
        template='plotly_white'
    )
    
    # Update all x and y axis labels
    fig.update_xaxes(title_text='K')
    fig.update_yaxes(title_text=y_label)
    
    fig.show()