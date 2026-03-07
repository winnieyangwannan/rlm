import pandas as pd
import os
import gzip
import json
import zlib
import numpy as np
import sys
import re
import math
from pathlib import Path

import ast
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import importlib

# Import categorize_error from invalid_submission_utils
_invalid_submission_utils_path = os.path.join(os.path.dirname(__file__), 'invalid_submission')
if _invalid_submission_utils_path not in sys.path:
    sys.path.insert(0, _invalid_submission_utils_path)
from invalid_submission_utils import categorize_error,  is_infra_error_category

# Import pass@k utility functions
_pass_at_k_utils_path = os.path.join(os.path.dirname(__file__), 'pass@k')
if _pass_at_k_utils_path not in sys.path:
    sys.path.insert(0, _pass_at_k_utils_path)
from pass_at_k_utils import (
    compute_max_g_at_k, 
    binary_pass_at_k_estimator,
    plot_pass_at_k_score,
    plot_average_pass_at_k_score,
    plot_pass_at_k_valid_submission,
)

# Try to import mlebench components
try:
    from mlebench.registry import registry
    MLEBENCH_AVAILABLE = True
except ImportError:
    registry = None
    MLEBENCH_AVAILABLE = False


MLE_30_7 = ["h-and-m-personalized-fashion-recommendations",
            "hotel-id-2021-fgvc8",
            "hubmap-kidney-segmentation",
            "kuzushiji-recognition",
            "mlsp-2013-birds",
            "spooky-author-identification",
            "whale-categorization-playground",]  

MLE_30 = [
    "aptos2019-blindness-detection",
    "h-and-m-personalized-fashion-recommendations",
    "mlsp-2013-birds",
    "plant-pathology-2021-fgvc8",
    "uw-madison-gi-tract-image-segmentation",
    "hms-harmful-brain-activity-classification",
    "multi-modal-gesture-recognition",
    "smartphone-decimeter-2022",
    "ventilator-pressure-prediction",
    "billion-word-imputation",
    "hotel-id-2021-fgvc8",
    "new-york-city-taxi-fare-prediction",
    "spooky-author-identification",
    "whale-categorization-playground",
    "bms-molecular-translation",
    "hubmap-kidney-segmentation",
    "nfl-player-contact-detection",
    "stanford-covid-vaccine",
    "cassava-leaf-disease-classification",
    "imet-2020-fgvc7",
    "nomad2018-predict-transparent-conductors",
    "tensorflow2-question-answering",
    "champs-scalar-coupling",
    "jigsaw-unintended-bias-in-toxicity-classification",
    "osic-pulmonary-fibrosis-progression",
    "tweet-sentiment-extraction",
    "freesound-audio-tagging-2019",
    "kuzushiji-recognition",
    "petfinder-pawpularity-score",
    "us-patent-phrase-to-phrase-matching"
    ]


def get_rank_and_percentile(score, leaderboard, lower_is_better):
    """
    Calculates the percentile rank of `score` as if it were an additional submission in the leaderboard.
    """
    if score is None or (isinstance(score, float) and np.isnan(score)):
        return {"percentile": 0, "rank": len(list(leaderboard["score"])) + 1}

    scores_list = list(leaderboard["score"]) + [score]
    n = len(scores_list)

    if lower_is_better:
        sorted_scores = sorted(scores_list)
    else:
        sorted_scores = sorted(scores_list, reverse=True)

    tol_rel = 1e-9
    tol_abs = 1e-12
    ranks = [
        i + 1
        for i, s in enumerate(sorted_scores)
        if math.isclose(s, score, rel_tol=tol_rel, abs_tol=tol_abs)
    ]

    if not ranks:
        ranks = [i + 1 for i, s in enumerate(sorted_scores) if s == score]

    avg_rank = sum(ranks) / len(ranks)
    percentile = (n - avg_rank) / (n - 1)

    return {"percentile": percentile, "rank": avg_rank}


def get_leaderboard(competition):
    """Get the leaderboard DataFrame for a competition."""
    return pd.read_csv(competition.leaderboard)

# Pattern to match special LLM tokens like <|eot_id|>, <|start_header_id|>, etc.
SPECIAL_TOKEN_PATTERN = re.compile(r'<\|.*?\|>')

# Pattern to collapse multiple newlines into a single newline
MULTI_NEWLINE_PATTERN = re.compile(r'\n{2,}')

# Pattern to add newline between </bash> and assistant
BASH_ASSISTANT_PATTERN = re.compile(r'</bash>assistant')

# Pattern to detect XML-like tags (including tags with attributes like <tool: bash>)
XML_TAG_PATTERN = re.compile(r'<(/?)(\w+)(?::\s*\w+)?(?:\s[^>]*)?>') 

# Pattern to match trailing orphaned closing tags (closing tags without matching opening tags at the end)
TRAILING_CLOSING_TAGS_PATTERN = re.compile(r'(</\w+>)+\s*$') 


def complete_xml_tags(text: str) -> str:
    """
    Complete incomplete XML tags in the text.
    
    If a closing tag appears without an opening tag, add the opening tag at the beginning.
    If an opening tag appears without a closing tag, add the closing tag at the end.
    
    Args:
        text: The text potentially containing incomplete XML tags
        
    Returns:
        Text with completed XML tags
    """
    if not text:
        return text
    
    # Track opening and closing tags
    # Stack for tracking open tags
    tag_stack = []
    # List of tags that need opening tags added at the beginning
    needs_opening = []
    
    # Find all tags in order
    for match in XML_TAG_PATTERN.finditer(text):
        is_closing = match.group(1) == '/'
        tag_name = match.group(2)
        
        if is_closing:
            # Check if there's a matching opening tag
            if tag_stack and tag_stack[-1] == tag_name:
                tag_stack.pop()
            else:
                # No matching opening tag, needs one at the beginning
                needs_opening.append(tag_name)
        else:
            # Opening tag
            tag_stack.append(tag_name)
    
    # Build the result
    result = text
    
    # Add missing opening tags at the beginning (in reverse order)
    for tag_name in reversed(needs_opening):
        result = f"<{tag_name}>" + result
    
    # Add missing closing tags at the end
    for tag_name in reversed(tag_stack):
        result = result + f"</{tag_name}>"
    
    return result


def strip_trailing_orphan_tags(text: str) -> str:
    """
    Strip trailing orphaned closing XML tags from the end of text.
    
    These are closing tags that appear at the very end without corresponding 
    opening tags in the text. This is common in truncated model outputs.
    
    Args:
        text: The text potentially containing trailing orphaned closing tags
        
    Returns:
        Text with trailing orphaned closing tags removed
    """
    if not text:
        return text
    
    # Keep stripping trailing closing tags until none remain
    while True:
        match = TRAILING_CLOSING_TAGS_PATTERN.search(text)
        if not match:
            break
        
        trailing_tags = match.group(0)
        # Extract tag names from the trailing tags
        tag_names = [m.group(1) for m in re.finditer(r'</(\w+)>', trailing_tags)]
        
        # Check if these closing tags have matching opening tags
        text_before_trailing = text[:match.start()]
        orphaned = False
        
        for tag_name in tag_names:
            # Count opening and closing tags for this tag name in text before trailing
            opening_count = len(re.findall(f'<{tag_name}(?:\\s[^>]*)?>(?!</)', text_before_trailing))
            closing_count = len(re.findall(f'</{tag_name}>', text_before_trailing))
            
            # If there are more closings than openings in trailing, they're orphaned
            if opening_count <= closing_count:
                orphaned = True
                break
        
        if orphaned:
            # Remove the trailing tags
            text = text_before_trailing.rstrip()
        else:
            break
    
    return text


def build_rollout(df_trj, rollout_id: int) -> list[dict]:
    """
    Build a rollout as a list of turn dicts for a given rollout_id.
    
    Args:
        df_trj: DataFrame containing trajectory data
        rollout_id: Index of the rollout in the DataFrame
        
    Returns: 
        List of dicts, each with "action" and "observation" fields
    """
    transitions = df_trj.iloc[rollout_id]["rollouts"][0]["traj"]["transitions"]
    
    rollout = []
    for t, turn in enumerate(transitions):
        rollout.append({
            "turn_id": t,
            "action": turn["action_str"],
            "observation": turn["observation_str"]
        })
    
    return rollout

def build_rollout_str(df_trj, rollout_id: int) -> str:
    """
    Build a rollout as a concatenated string for a given rollout_id.
    
    Args:
        df_trj: DataFrame containing trajectory data
        rollout_id: Index of the rollout in the DataFrame
        
    Returns: 
        A string with all turns concatenated, each with turn_id, action, and observation.
    """
    transitions = df_trj.iloc[rollout_id]["rollouts"][0]["traj"]["transitions"]
    
    rollout_parts = []
    for t, turn in enumerate(transitions):
        step = t + 1  # 1-indexed steps
        # Complete XML tags in action_str
        action = complete_xml_tags(turn['action_str'])
        # Clean special tokens from observation and normalize whitespace
        observation = SPECIAL_TOKEN_PATTERN.sub('', turn['observation_str'])
        observation = BASH_ASSISTANT_PATTERN.sub('</bash>\n\nassistant', observation)
        observation = MULTI_NEWLINE_PATTERN.sub('\n', observation).strip()
        rollout_parts.append(
            f"Step {step} - Agent Action:\n{action}\n\n"
            f"Step {step} - Environment Response:\n{observation}"
        )
    
    return "\n\n".join(rollout_parts)


def print_rollout(rollout_str: str, action_color: str = "blue", observation_color: str = "green", show_think: bool = True):
    """
    Print rollout_str in a human-readable format with colored sections and separators.
    
    Args:
        rollout_str: The rollout string from flatten_dataframe's rollout_str column
        action_color: Color for agent action sections (default: blue)
                      Options: black, red, green, yellow, blue, magenta, cyan, white
        observation_color: Color for environment observation sections (default: green)
        show_think: Whether to display <think>...</think> content (default: True). 
                   If True, think content is shown in gray italic style.
    """
    from IPython.display import display, HTML
    import html
    import re
    
    # Strip trailing orphaned closing tags instead of completing them
    rollout_str = strip_trailing_orphan_tags(rollout_str)
    
    def escape_content(text: str) -> str:
        """Escape HTML/XML tags so they display as literal text."""
        return html.escape(text)
    
    def fix_malformed_think_tags(text: str) -> str:
        """Fix malformed think tag patterns in text.
        
        Handles:
        1. Empty <think></think> followed by content before <tool: - moves content into tags
        2. Unclosed <think> at end of text - adds closing </think>
        """
        # Pattern 1: Empty <think></think> followed by content before <tool:
        # This captures: <think></think>\n\nSome thinking text...\n\n<tool: bash>
        empty_think_pattern = r'<think>\s*</think>\s*\n+(.*?)(?=\n*<tool:|$)'
        
        def replace_empty_think(match):
            content = match.group(1).strip()
            if content:
                return f'<think>\n{content}\n</think>'
            return '<think></think>'
        
        text = re.sub(empty_think_pattern, replace_empty_think, text, flags=re.DOTALL)
        
        # Pattern 2: Unclosed <think> at end - find <think> without matching </think>
        # Count opening and closing think tags
        open_thinks = list(re.finditer(r'<think>', text))
        close_thinks = list(re.finditer(r'</think>', text))
        
        if len(open_thinks) > len(close_thinks):
            # Find the last unclosed <think>
            last_open_pos = open_thinks[-1].end()
            # Add closing tag at the end
            text = text.rstrip() + '\n</think>'
        
        return text
    
    def format_action_with_think(text: str, show_think: bool) -> str:
        """Format action text, handling <think>...</think> tags specially.
        
        Also handles malformed cases:
        - Empty <think></think> followed by thinking content before <tool:
        - Unclosed <think> tags at the end of text
        """
        # First, fix malformed think tags
        text = fix_malformed_think_tags(text)
        
        # Pattern to match <think>...</think> including empty tags and multiline content
        think_pattern = r'<think>(.*?)</think>'
        
        if not show_think:
            # Remove think tags and their content entirely
            text = re.sub(think_pattern, '', text, flags=re.DOTALL)
            # Clean up extra newlines left behind
            text = re.sub(r'\n{3,}', '\n\n', text)
            return escape_content(text.strip())
        
        # Split text by think tags and format each part
        parts = []
        last_end = 0
        
        for match in re.finditer(think_pattern, text, flags=re.DOTALL):
            # Add text before think tag
            before_text = text[last_end:match.start()]
            if before_text:
                parts.append(escape_content(before_text))
            
            # Add think content with special styling
            think_content = match.group(1).strip()
            if think_content:
                escaped_think = escape_content(think_content)
                parts.append(f'<span style="color: #7f8c8d; font-style: italic; background-color: #f5f5f5; padding: 2px 4px; border-radius: 3px;">&lt;think&gt;{escaped_think}&lt;/think&gt;</span>')
            else:
                # Empty think tags - show them but grayed out
                parts.append('<span style="color: #bdc3c7; font-style: italic;">&lt;think&gt;&lt;/think&gt;</span>')
            
            last_end = match.end()
        
        # Add remaining text after last think tag
        remaining = text[last_end:]
        if remaining:
            parts.append(escape_content(remaining))
        
        return ''.join(parts) if parts else escape_content(text)
    
    # ANSI color codes for terminal
    colors = {
        'black': '\033[30m',
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'reset': '\033[0m',
        'bold': '\033[1m',
    }
    
    # HTML colors for Jupyter
    html_colors = {
        'black': '#000000',
        'red': '#e74c3c',
        'green': '#27ae60',
        'yellow': '#f39c12',
        'blue': '#3498db',
        'magenta': '#9b59b6',
        'cyan': '#1abc9c',
        'white': '#ecf0f1',
    }
    
    separator = "=" * 80
    
    # Split by steps
    lines = rollout_str.split('\n')
    
    html_output = []
    current_section = None
    current_content = []
    
    for line in lines:
        if line.startswith("Step ") and " - Agent Action:" in line:
            # Output previous section if exists
            if current_section and current_content:
                color = html_colors.get(observation_color, html_colors['green'])
                content = escape_content('\n'.join(current_content))
                html_output.append(f'<pre style="color: {color}; background-color: #f8f9fa; padding: 10px; border-radius: 5px; overflow-x: auto; white-space: pre-wrap;">{content}</pre>')
            
            # Start new action section
            html_output.append(f'<div style="background-color: #2c3e50; color: white; padding: 8px; margin-top: 15px; border-radius: 5px 5px 0 0;"><strong>{escape_content(line)}</strong></div>')
            current_section = 'action'
            current_content = []
            
        elif line.startswith("Step ") and " - Environment Response:" in line:
            # Output previous action section
            if current_section == 'action' and current_content:
                color = html_colors.get(action_color, html_colors['blue'])
                content = format_action_with_think('\n'.join(current_content), show_think)
                html_output.append(f'<pre style="color: {color}; background-color: #eef6fc; padding: 10px; border-radius: 0 0 5px 5px; overflow-x: auto; white-space: pre-wrap;">{content}</pre>')
            
            # Start new observation section
            html_output.append(f'<div style="background-color: #1e8449; color: white; padding: 8px; margin-top: 10px; border-radius: 5px 5px 0 0;"><strong>{escape_content(line)}</strong></div>')
            current_section = 'observation'
            current_content = []
            
        else:
            current_content.append(line)
    
    # Output final section
    if current_section and current_content:
        if current_section == 'action':
            color = html_colors.get(action_color, html_colors['blue'])
            bg_color = '#eef6fc'
            content = format_action_with_think('\n'.join(current_content), show_think)
        else:
            color = html_colors.get(observation_color, html_colors['green'])
            bg_color = '#f8f9fa'
            content = escape_content('\n'.join(current_content))
        html_output.append(f'<pre style="color: {color}; background-color: {bg_color}; padding: 10px; border-radius: 0 0 5px 5px; overflow-x: auto; white-space: pre-wrap;">{content}</pre>')
    
    # Display as HTML in Jupyter
    display(HTML(''.join(html_output)))


def calculate_rollout_tokens(df_trj, rollout_id: int) -> tuple[int, int]:
    """
    Calculate total tokens in a rollout.
    
    Args:
        df_trj: DataFrame containing trajectory data
        rollout_id: Index of the rollout in the DataFrame
        
    Returns: 
        Tuple of (total_rollout_tokens, total_action_tokens)
        - total_rollout_tokens: sum of len(action) + len(observation) for all turns
        - total_action_tokens: sum of len(action) for all turns
    """
    transitions = df_trj.iloc[rollout_id]["rollouts"][0]["traj"]["transitions"]
    
    total_rollout_tokens = 0
    total_action_tokens = 0
    
    for turn in transitions:
        action_tokens = len(turn["action"])
        observation_tokens = len(turn["observation"])
        total_action_tokens += action_tokens
        total_rollout_tokens += action_tokens + observation_tokens
    
    return total_rollout_tokens, total_action_tokens

def get_medal_from_percentile(
    task_name: str,
    percentile: float,
    mle_bench_data_dir: str = "/checkpoint/maui/shared/cache/dojo/tasks/mlebench"
) -> str:
    """
    Determine the medal earned for a given percentile score.
    
    Args:
        task_name: Name of the competition/task
        percentile: The percentile score achieved
        mle_bench_data_dir: Path to MLE bench data directory
        
    Returns:
        "gold", "silver", "bronze", or "" (empty string if no medal)
    """
    if not MLEBENCH_AVAILABLE:
        return ""
    
    if percentile is None or (isinstance(percentile, float) and np.isnan(percentile)):
        return ""
    
    try:
        new_registry = registry.set_data_dir(Path(mle_bench_data_dir))
        competition = new_registry.get_competition(task_name)
        competition_leaderboard = get_leaderboard(competition)
        rank_info = competition.grader.rank_score(0, competition_leaderboard)
        is_lower_better = competition.grader.is_lower_better(competition_leaderboard)
        
        # Get medal thresholds as percentiles
        gold_threshold = get_rank_and_percentile(
            rank_info["gold_threshold"], competition_leaderboard, is_lower_better
        )["percentile"]
        silver_threshold = get_rank_and_percentile(
            rank_info["silver_threshold"], competition_leaderboard, is_lower_better
        )["percentile"]
        bronze_threshold = get_rank_and_percentile(
            rank_info["bronze_threshold"], competition_leaderboard, is_lower_better
        )["percentile"]
        
        # Determine medal
        if percentile >= gold_threshold:
            return "gold"
        elif percentile >= silver_threshold:
            return "silver"
        elif percentile >= bronze_threshold:
            return "bronze"
        else:
            return ""
            
    except Exception as e:
        print(f"Error getting medal for task {task_name}: {e}")
        return ""


def remove_infra_error_rows(df_trj, infra_error_indices, trajectory_path=None, save=False):
    """
    Create a new dataframe without rows at the specified infrastructure error indices.
    Also loads and filters the all_metrics.jsonl file if trajectory_path is provided.
    
    Args:
        df_trj: Original trajectory dataframe
        infra_error_indices: List of indices to remove (rows with infrastructure errors)
        trajectory_path: Path to the trajectory jsonl file. If provided, also loads and filters
                        all_metrics.jsonl from the trajectories directory.
                        Expected format: .../trajectories/<subfolder>/<file>.jsonl
                        all_metrics.jsonl is at: .../trajectories/all_metrics.jsonl
        save: If True, saves the cleaned dataframes back to their original file paths
    
    Returns:
        df_trj_clean: Cleaned dataframe with infra error rows removed
        df_metrics_clean: Cleaned metrics dataframe (only if trajectory_path is provided)
    """
    if not infra_error_indices:
        infra_error_indices = []
    
    # Create mask to exclude infra error indices
    mask = ~df_trj.index.isin(infra_error_indices)
    df_trj_clean = df_trj[mask].reset_index(drop=True)
    
    print(f"Original rows: {len(df_trj)}")
    print(f"Removed {len(infra_error_indices)} infrastructure error rows")
    print(f"Cleaned rows: {len(df_trj_clean)}")
    print(f"df_trj_clean shape: {df_trj_clean.shape}")
    
    # Save df_trj_clean if requested
    if save and trajectory_path:
        df_trj_clean.to_json(trajectory_path, orient='records', lines=True)
        print(f"\nSaved df_trj_clean to: {trajectory_path}")
    
    # Load and filter all_metrics.jsonl if trajectory_path is provided
    if trajectory_path:
        # trajectory_path: .../trajectories/<subfolder>/<file>.jsonl
        # all_metrics.jsonl is in: .../trajectories/all_metrics.jsonl
        # So we need to go up two levels from the file
        trajectory_subdir = os.path.dirname(trajectory_path)  # .../trajectories/<subfolder>
        trajectories_dir = os.path.dirname(trajectory_subdir)  # .../trajectories
        metrics_path = os.path.join(trajectories_dir, "all_metrics.jsonl")
        
        if os.path.exists(metrics_path):
            df_metrics = pd.read_json(metrics_path, lines=True)
            print(f"\nLoaded metrics file from: {metrics_path}")
            print(f"Metrics file has {len(df_metrics)} rows")
            
            # Apply same filtering
            mask_metrics = ~df_metrics.index.isin(infra_error_indices)
            df_metrics_clean = df_metrics[mask_metrics].reset_index(drop=True)
            print(f"Cleaned metrics rows: {len(df_metrics_clean)}")
            print(f"df_metrics_clean shape: {df_metrics_clean.shape}")
            
            # Save df_metrics_clean if requested
            if save:
                df_metrics_clean.to_json(metrics_path, orient='records', lines=True)
                print(f"Saved df_metrics_clean to: {metrics_path}")
            
            return df_trj_clean, df_metrics_clean
        else:
            print(f"\nWarning: Metrics file not found at {metrics_path}")
            return df_trj_clean, None
    
    return df_trj_clean


def flatten_dataframe(
    df_trj,
    only_valid_submissions: bool = False,
    mle_bench_data_dir: str = "/checkpoint/maui_sft/winnieyangwn/datasets",
    filter_out_errors: list[str] | str | None = [
        "Too many streaming blocks of output",
        "Infrastructure: Connection/Heartbeat Error",
        "Infrastructure: AgentBox Backend Not Initialized",
        "Infrastructure: Container Error",
        "Infrastructure: Container Execution Error",
        "Infrastructure: Container Creation Error",
        "Infrastructure: Model Call Error",
    ],
    pass_at_k_values: list[int] | None = [1, 2, 4, 8, 16, 24, 32, 40, 48],
    return_infra_error_indices: bool = False,
    save: bool = False,
    save_path: str | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, list[int]]:
    """
    Flatten the trajectory DataFrame into a simplified pandas DataFrame.
    
    Args:
        df_trj: DataFrame containing trajectory data
        only_valid_submissions: If True, only keep rollouts with valid_submission=True.
            Defaults to False.
        mle_bench_data_dir: Path to MLE bench data directory for medal calculation.
        filter_out_errors: Optional error category or list of categories to filter out.
            Uses the same categorization as analyze_invalid_submissions.
            Example: "Too many streaming blocks of output" or 
            ["Too many streaming blocks of output", "Timeout"]
        pass_at_k_values: List of K values for computing pass@K metrics.
            Defaults to [1, 2, 4, 8, 16, 32, 64]. Set to None to skip pass@K computation.
        return_infra_error_indices: If True, also return a list of indices in df_trj
            that have infrastructure errors. Defaults to False.
        save: If True, save the flattened DataFrame as a .jsonl file. Defaults to False.
        save_path: Path to save the .jsonl file. Required if save=True.
    
    Returns:
        If return_infra_error_indices is False:
            DataFrame with task_name, task_description, code, percentile, medal, valid_submission,
            eval_error_output, eval_duration, rollout_duration, num_turns, rollout, rollout_str,
            container_execution_error, row_indx columns, plus pass@K_percentile and pass@K_valid
            columns for each K in pass_at_k_values.
        
        If return_infra_error_indices is True:
            Tuple of (DataFrame, list of infrastructure error indices)
        
        Also includes from outcomes: test_timeout, max_turns_reached, test_execution_error,
        container_creation_error, model_call_error, rollout_timeout.
        
        Also includes from info: duration_per_turn, info_rollout_duration, action_tokens_per_turn,
        action_tokens_rollout, observation_tokens_per_turn, observation_tokens_rollout,
        context_tokens_rollout.
    """
    # Normalize filter_out_errors to a set for efficient lookup
    if filter_out_errors is None:
        filter_out_errors_set = set()
    elif isinstance(filter_out_errors, str):
        filter_out_errors_set = {filter_out_errors}
    else:
        filter_out_errors_set = set(filter_out_errors)
    
    # Use the shared INFRA_ERROR_CATEGORIES constant from invalid_submission_utils
    # This ensures consistency with categorize_error and plot_submission_validity_breakdown
    
    rows = []
    infra_error_indices = []  # Track indices with infrastructure errors
    
    for idx in range(len(df_trj)):
        row = df_trj.iloc[idx]
        rollout_data = row["rollouts"][0]
        transitions = rollout_data["traj"]["transitions"]
        last_outcomes = transitions[-1]["outcomes"]
        info = transitions[-1]["info"]
        info_output = info.get("output")
        observation_strs = "\n".join(t.get("observation_str", "") for t in transitions)
        
        valid_submission = last_outcomes.get("valid_submission")
        
        if only_valid_submissions and not valid_submission:
            continue
        
        # Check for container execution error or agentbox backend error
        container_execution_error = last_outcomes.get("container_execution_error", False)
        container_creation_error = last_outcomes.get("container_creation_error", False)
        model_call_error = last_outcomes.get("model_call_error", False)
        max_turns_reached = last_outcomes.get("max_turns_reached", False)
        parse_error = last_outcomes.get("parse_error", False)
        info_str = str(info_output).lower() if info_output else ""
        if "agentboxbackend" in info_str and "has no attribute" in info_str and "container" in info_str:
            container_execution_error = True
        
        # Categorize error for this row (used for filtering and tracking infra errors)
        error_output = last_outcomes.get("eval_error_output", "")
        error_category = categorize_error({
            "eval_error_output": error_output,
            "info_output": info_output,
            "rollout_str": observation_strs,
            "container_execution_error": container_execution_error,
            "container_creation_error": container_creation_error,
            "model_call_error": model_call_error,
            "max_turns_reached": max_turns_reached,
            "parse_error": parse_error,
        })
        
        # Track infrastructure error indices regardless of filtering
        # Use the shared is_infra_error_category function for consistency
        if is_infra_error_category(error_category):
            infra_error_indices.append(idx)
        
        # Filter out specified error categories if requested
        if filter_out_errors_set and not valid_submission:
            if error_category in filter_out_errors_set:
                continue
        
        task_name = rollout_data["start_args"]["instance_id"]
        percentile = last_outcomes.get("percentile")
        medal = get_medal_from_percentile(task_name, percentile, mle_bench_data_dir)
        
        rows.append({
            # "row_indx": idx,
            "task_name": task_name,
            "task_description": rollout_data["start_args"]["task_description"],
            "code": info.get("pred_solution"),
            "percentile": percentile,
            "medal": medal,
            "valid_submission": valid_submission,
            "eval_error_output": last_outcomes.get("eval_error_output"),
            "eval_duration": last_outcomes.get("gpu_execution_duration"),
            "rollout_duration": rollout_data["metrics"].get("rollout/duration"),
            "num_turns": len(transitions),
            "rollout": build_rollout(df_trj, idx),
            "rollout_str": build_rollout_str(df_trj, idx),
            "container_execution_error": container_execution_error,
            # Additional outcomes fields
            "test_timeout": last_outcomes.get("test_timeout"),
            "max_turns_reached": max_turns_reached,
            "test_execution_error": last_outcomes.get("test_execution_error"),
            "container_creation_error": container_creation_error,
            "model_call_error": model_call_error,
            "rollout_timeout": last_outcomes.get("rollout_timeout"),
            "pred_solution_provided": last_outcomes.get("pred_solution_provided"),
            # Additional info fields
            "duration_per_turn": info.get("duration_per_turn"),
            "info_rollout_duration": info.get("rollout_duration"),
            "action_tokens_per_turn": info.get("action_tokens_per_turn"),
            "action_tokens_rollout": info.get("action_tokens_rollout"),
            "observation_tokens_per_turn": info.get("observation_tokens_per_turn"),
            "observation_tokens_rollout": info.get("observation_tokens_rollout"),
            "context_tokens_rollout": info.get("context_tokens_rollout"),
        })
    
    result_df = pd.DataFrame(rows)
    
    # Preserve original df_trj index even after filtering
    if len(result_df) > 0 and "row_indx" in result_df.columns:
        result_df.index = result_df["row_indx"]
    
    # Compute pass@K values if requested
    if pass_at_k_values and len(result_df) > 0:
        result_df = _add_pass_at_k_columns(result_df, pass_at_k_values)
    
    # Save to .jsonl file if requested
    if save:
        if save_path is None:
            raise ValueError("save_path must be provided when save=True")
        result_df.to_json(save_path, orient="records", lines=True)
        print(f"Saved flattened DataFrame to {save_path}")
    
    if return_infra_error_indices:
        return result_df, infra_error_indices
    return result_df


def _add_pass_at_k_columns(df: pd.DataFrame, k_values: list[int]) -> pd.DataFrame:
    """
    Add pass@K columns to the flattened DataFrame.
    
    Computes pass@K for both percentile (continuous) and valid_submission (binary)
    for each task, then merges the results back to the DataFrame.
    
    Args:
        df: Flattened DataFrame with task_name, percentile, and valid_submission columns
        k_values: List of K values to compute pass@K for
        
    Returns:
        DataFrame with added pass@K_percentile and pass@K_valid columns for each K
    """
    # Group percentile and valid_submission by task
    percentiles_by_task = df.groupby("task_name")["percentile"].apply(list)
    valid_by_task = df.groupby("task_name")["valid_submission"].apply(list)
    
    # Compute pass@K for each task and K value
    pass_at_k_results = {}
    
    for K in k_values:
        pass_at_k_results[f"pass@{K}_percentile"] = {}
        pass_at_k_results[f"pass@{K}_valid"] = {}
        
        for task_name in percentiles_by_task.index:
            # Percentile pass@K (continuous estimator)
            percentiles = np.array(percentiles_by_task[task_name])
            # Handle NaN values - filter them out for computation
            valid_percentiles = percentiles[~np.isnan(percentiles.astype(float))]
            if len(valid_percentiles) >= K:
                estimate = compute_max_g_at_k(valid_percentiles, K=K)
                pass_at_k_results[f"pass@{K}_percentile"][task_name] = estimate
            else:
                pass_at_k_results[f"pass@{K}_percentile"][task_name] = np.nan
            
            # Valid submission pass@K (binary estimator)
            valid_submissions = np.array(valid_by_task[task_name])
            # Filter out None values and convert to bool
            valid_submissions_clean = [v for v in valid_submissions if v is not None]
            n = len(valid_submissions_clean)
            if n >= K:
                c = int(sum(bool(v) for v in valid_submissions_clean))  # count of valid submissions
                estimate = binary_pass_at_k_estimator(n, c, K)
                pass_at_k_results[f"pass@{K}_valid"][task_name] = estimate
            else:
                pass_at_k_results[f"pass@{K}_valid"][task_name] = np.nan
    
    # Create a DataFrame with pass@K results indexed by task_name
    pass_at_k_df = pd.DataFrame(pass_at_k_results)
    pass_at_k_df.index.name = "task_name"
    pass_at_k_df = pass_at_k_df.reset_index()
    
    # Merge pass@K columns back to the original DataFrame
    df = df.merge(pass_at_k_df, on="task_name", how="left")
    
    return df


def get_medal_info(df: pd.DataFrame, task_name: str) -> list[tuple[int, str]]:
    """
    Extract the row indices of rollouts with a medal for a given task.
    
    Args:
        df: DataFrame with 'task_name' and 'medal' columns
        task_name: Name of the task to filter by
        
    Returns:
        List of tuples (row_index, medal_type) where medal_type is "gold", "silver", or "bronze".
    """
    mask = (df["task_name"] == task_name) & (df["medal"].isin(["gold", "silver", "bronze"]))
    filtered = df[mask]
    return [(idx, row["medal"]) for idx, row in filtered.iterrows()]


def plot_percentile_histograms_by_task(df, task_names=None, n_bins=20, n_cols=3, height_per_row=300):
    """
    Plot interactive histogram of percentile scores for each task using Plotly.
    
    Args:
        df: DataFrame with 'task_name' and 'percentile' columns
        task_names: Optional list of task names to plot. If None, plot all tasks.
        n_bins: Number of bins between 0 and 1 (default: 20)
        n_cols: Number of columns in subplot grid (default: 3)
        height_per_row: Height per row in pixels (default: 300)
    """
    if task_names is not None:
        tasks = [t for t in task_names if t in df["task_name"].values]
    else:
        tasks = sorted(df["task_name"].unique())
    
    n_tasks = len(tasks)
    n_rows = (n_tasks + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=tasks,
        horizontal_spacing=0.05,
        vertical_spacing=0.1
    )
    
    bins_edges = np.linspace(0, 1, n_bins + 1)
    
    for i, task in enumerate(tasks):
        row = i // n_cols + 1
        col = i % n_cols + 1
        task_data = df[df["task_name"] == task]["percentile"]
        
        fig.add_trace(
            go.Histogram(
                x=task_data,
                xbins=dict(start=0, end=1, size=1/n_bins),
                name=task,
                showlegend=False,
                opacity=0.7
            ),
            row=row, col=col
        )
        fig.update_xaxes(range=[0, 1], title_text="Percentile", row=row, col=col)
        fig.update_yaxes(title_text="Count", row=row, col=col)
    
    fig.update_layout(
        height=height_per_row * n_rows,
        width=1200,
        title_text="Percentile Distribution by Task"
    )
    fig.show()


def extract_pass_at_k_df(df: pd.DataFrame, metric: str = "percentile") -> pd.DataFrame:
    """
    Extract pass@K values from a flattened DataFrame into a format suitable for plotting.
    
    Args:
        df: Flattened DataFrame from flatten_dataframe with pass@K columns
        metric: "percentile" or "valid" to extract the corresponding pass@K values
        
    Returns:
        DataFrame indexed by task_name with pass@K columns (e.g., pass@1, pass@2, ...)
    """
    # Find pass@K columns for the specified metric
    suffix = f"_{metric}"
    pass_at_k_cols = [col for col in df.columns if col.startswith("pass@") and col.endswith(suffix)]
    
    if not pass_at_k_cols:
        raise ValueError(f"No pass@K columns found for metric '{metric}'. "
                        f"Available columns: {[c for c in df.columns if c.startswith('pass@')]}")
    
    # Extract unique task pass@K values (they're the same for all rows of a task)
    result = df.groupby("task_name")[pass_at_k_cols].first()
    
    # Rename columns to remove the suffix (e.g., pass@1_percentile -> pass@1)
    result.columns = [col.replace(suffix, "") for col in result.columns]
    
    return result


def plot_pass_at_k(
    dfs: pd.DataFrame | list[pd.DataFrame],
    metric: str = "percentile",
    labels: list[str] | None = None,
    task_names: list[str] | None = None,
    Ks: list[int] | None = None,
    colormap: str | None = "YlGnBu",
):
    """
    Plot pass@K results from flattened DataFrames using interactive Plotly plots.
    
    Args:
        dfs: Single DataFrame or list of DataFrames from flatten_dataframe
        metric: "percentile" or "valid" - which metric to plot
        labels: Optional list of labels for legend (same length as dfs)
        task_names: Optional list of task names to plot. If None, plots all tasks.
        Ks: Optional list of K values to use. If None, auto-detect from columns.
        colormap: Optional matplotlib colormap name (e.g., "Greens", "Blues", "Reds").
            If None, uses default discrete colors.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Normalize to list
    if isinstance(dfs, pd.DataFrame):
        dfs = [dfs]
    
    # Default labels
    if labels is None:
        labels = [f"Model {i+1}" for i in range(len(dfs))]
    
    if len(labels) != len(dfs):
        raise ValueError(f"labels length ({len(labels)}) must match dfs length ({len(dfs)})")
    
    # Extract pass@K DataFrames for all inputs
    results_list = [extract_pass_at_k_df(df, metric=metric) for df in dfs]
    
    # Auto-detect K values if not provided
    if Ks is None:
        Ks = sorted([int(col.replace("pass@", "")) for col in results_list[0].columns])
    
    # Get all tasks (union across all DataFrames)
    all_tasks = set()
    for result in results_list:
        all_tasks.update(result.index.tolist())
    all_tasks = sorted(all_tasks)
    
    # Filter to specified task names if provided
    if task_names is not None:
        all_tasks = [t for t in task_names if t in all_tasks]
    
    n_tasks = len(all_tasks)
    
    # Define colors for multiple models
    if colormap is not None:
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap(colormap)
        n_models = len(dfs)
        # Sample colors from colormap, avoiding very light colors (start from 0.3)
        colors = [f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})' 
                  for c in [cmap(0.3 + 0.7 * i / max(n_models - 1, 1)) for i in range(n_models)]]
    else:
        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']
    symbols = ['circle', 'square', 'triangle-up', 'diamond', 'triangle-down', 'star', 'hexagon', 'cross']
    
    # Calculate grid size for subplots
    n_cols = 4
    n_rows = (n_tasks + n_cols - 1) // n_cols
    
    # Calculate spacing that respects Plotly's constraints
    # vertical_spacing must be < 1/(rows-1), horizontal_spacing must be < 1/(cols-1)
    max_v_spacing = 1.0 / (n_rows - 1) if n_rows > 1 else 0.03
    max_h_spacing = 1.0 / (n_cols - 1) if n_cols > 1 else 0.03
    vertical_spacing = min(0.03, max_v_spacing * 0.7)  # Use smaller spacing for compact layout
    horizontal_spacing = min(0.03, max_h_spacing * 0.7)
    
    # Plot per-task results using Plotly subplots
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=all_tasks,
        horizontal_spacing=horizontal_spacing,
        vertical_spacing=vertical_spacing
    )
    
    ylabel = 'Percentile Score' if metric == "percentile" else 'Valid Submission Rate'
    
    # Track which labels have already been shown in the legend
    legend_shown = set()
    
    for i, task_name in enumerate(all_tasks):
        row = i // n_cols + 1
        col = i % n_cols + 1
        
        for j, (result, label) in enumerate(zip(results_list, labels)):
            if task_name in result.index:
                values = [result.loc[task_name, f"pass@{K}"] if f"pass@{K}" in result.columns else np.nan for K in Ks]
                # Only show legend for first occurrence of each label to avoid duplicates
                show_legend = label not in legend_shown
                if show_legend:
                    legend_shown.add(label)
                fig.add_trace(
                    go.Scatter(
                        x=Ks, y=values,
                        mode='lines+markers',
                        name=label,
                        line=dict(color=colors[j % len(colors)]),
                        marker=dict(symbol=symbols[j % len(symbols)], size=6),
                        legendgroup=label,
                        showlegend=show_legend,
                        hovertemplate=f"{label}<br>K=%{{x}}<br>{ylabel}=%{{y:.4f}}<extra></extra>"
                    ),
                    row=row, col=col
                )
    
    # Update layout for per-task figure - use square subplots
    subplot_size = 350
    fig.update_xaxes(title_text='K')
    fig.update_yaxes(range=[-0.1, 1.1])
    fig.update_layout(
        height=subplot_size * n_rows,
        width=subplot_size * n_cols,
        title_text=f'Pass@K {ylabel} by Task',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    # Make subplot titles smaller
    fig.update_annotations(font_size=9)
    fig.show()
    
    # Plot average results (only for filtered tasks)
    fig_avg = go.Figure()
    
    for j, (result, label) in enumerate(zip(results_list, labels)):
        # Filter result to only include tasks in all_tasks
        filtered_result = result.loc[result.index.intersection(all_tasks)]
        avg_values = [filtered_result[f"pass@{K}"].mean() if f"pass@{K}" in filtered_result.columns else np.nan for K in Ks]
        fig_avg.add_trace(
            go.Scatter(
                x=Ks, y=avg_values,
                mode='lines+markers',
                name=label,
                line=dict(color=colors[j % len(colors)], width=2),
                marker=dict(symbol=symbols[j % len(symbols)], size=8),
                hovertemplate=f"{label}<br>K=%{{x}}<br>Avg {ylabel}=%{{y:.4f}}<extra></extra>"
            )
        )
    
    avg_ylabel = 'Percentile' if metric == "percentile" else 'Valid Submission Rate'
    
    fig_avg.update_layout(
        title=f'pass@k {"Percentile" if metric == "percentile" else "Valid Submission"}',
        xaxis_title='K',
        yaxis_title=avg_ylabel,
        yaxis=dict(range=[0, 1]),
        height=500,
        width=700,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='x unified'
    )
    fig_avg.show()
    
    # Print the values
    print(f"K\t" + "\t".join(labels))
    for k_idx, K in enumerate(Ks):
        values = []
        for result in results_list:
            filtered_result = result.loc[result.index.intersection(all_tasks)]
            if f"pass@{K}" in filtered_result.columns:
                values.append(f"{filtered_result[f'pass@{K}'].mean():.4f}")
            else:
                values.append("N/A")
        print(f"{K}\t" + "\t".join(values))

