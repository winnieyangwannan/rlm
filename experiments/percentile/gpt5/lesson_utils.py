from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from dotenv import load_dotenv

from rlm import RLM
from rlm.logger import RLMLogger

if TYPE_CHECKING:
    from rlm.core.types import RLMChatCompletion
    
def load_summarization_folder(summarization_dir: str, task_name: str) -> Path | None:
    """Find summarization folder for a given task.
    
    Args:
        summarization_dir: Base directory containing task subfolders with .md files
        task_name: Task name to find the folder for
        
    Returns:
        Path to the task folder containing .md files, or None if not found
    """
    # Task folder is at summarization_dir/task_name/
    task_folder = Path(summarization_dir) / task_name
    
    if not task_folder.exists():
        print(f"Task folder not found: {task_folder}")
        return None
    
    # Count .md files in the folder
    md_files = list(task_folder.glob("*.md"))
    
    if not md_files:
        print(f"No .md files found in task folder: {task_folder}")
        return None
    
    print(f"Found {len(md_files)} .md files in: {task_folder}")
    return task_folder


def assemble_summary(
    data: dict,
    rollout_info: dict,
    task_name: str,
    tier: str,
    flatten_fn: callable,
) -> str:
    """Assemble a formatted markdown summary from raw JSON data.
    
    Args:
        data: Raw JSON dict from the .md file
        rollout_info: Dict with rollout_id, score_percentile, medal_earned
        task_name: Task name string
        tier: Pre-computed tier label
        flatten_fn: Function to flatten dict fields to string
    
    Returns:
        Formatted markdown string with all fields, with visual separators
        between sections and blank lines between subfields for readability.
    """
    def format_subfields(content: str) -> str:
        """Add blank lines between subfields (lines starting with field names)."""
        if not content:
            return content
        lines = content.split("\n")
        formatted_lines = []
        for i, line in enumerate(lines):
            # Add blank line before subfield headers (lines ending with ":")
            # but not for the first line or indented continuation lines
            if i > 0 and line and not line.startswith(" ") and ":" in line:
                formatted_lines.append("")
            formatted_lines.append(line)
        return "\n".join(formatted_lines)
    
    parts = []
    
    # Header with rollout info
    rollout_id = rollout_info.get("rollout_id", "N/A")
    percentile = rollout_info.get("score_percentile", "N/A")
    medal = rollout_info.get("medal_earned", None)
    
    task_str = f"# Task: {task_name}" if task_name else "# Task: N/A"
    tier_str = f" | Tier: {tier}" if tier else ""
    medal_str = f" | Medal: {medal if medal else 'None'}"
    if isinstance(percentile, (int, float)):
        parts.append(f"{task_str} | Rollout {rollout_id} : Percentile: {percentile:.2f}{tier_str}{medal_str}")
    else:
        parts.append(f"{task_str} | Rollout {rollout_id} : Percentile: {percentile}{tier_str}{medal_str}")
    parts.append("")
    parts.append("---")
    parts.append("")
    
    # Summary
    if data.get("summary"):
        parts.append("## Summary")
        parts.append("")
        parts.append(data["summary"])
        parts.append("")
        parts.append("---")
        parts.append("")
    
    # Task Analysis
    task_analysis = flatten_fn(data.get("task_analysis"))
    if task_analysis:
        parts.append("## Task Analysis")
        parts.append("")
        parts.append(format_subfields(task_analysis))
        parts.append("")
        parts.append("---")
        parts.append("")
    
    # Data Preprocessing
    data_preprocessing = flatten_fn(data.get("data_preprocessing"))
    if data_preprocessing:
        parts.append("## Data Preprocessing")
        parts.append("")
        parts.append(format_subfields(data_preprocessing))
        parts.append("")
        parts.append("---")
        parts.append("")
    
    # Feature Engineering
    feature_engineering = flatten_fn(data.get("feature_engineering"))
    if feature_engineering:
        parts.append("## Feature Engineering")
        parts.append("")
        parts.append(format_subfields(feature_engineering))
        parts.append("")
        parts.append("---")
        parts.append("")
    
    # Model Selection
    model_selection = flatten_fn(data.get("model_selection"))
    if model_selection:
        parts.append("## Model Selection")
        parts.append("")
        parts.append(format_subfields(model_selection))
        parts.append("")
        parts.append("---")
        parts.append("")
    
    # Training Methodology
    training_methodology = flatten_fn(data.get("training_methodology"))
    if training_methodology:
        parts.append("## Training Methodology")
        parts.append("")
        parts.append(format_subfields(training_methodology))
        parts.append("")
        parts.append("---")
        parts.append("")
    
    # Evaluation and Submission
    evaluation_and_submission = flatten_fn(data.get("evaluation_and_submission"))
    if evaluation_and_submission:
        parts.append("## Evaluation and Submission")
        parts.append("")
        parts.append(format_subfields(evaluation_and_submission))
        parts.append("")
        parts.append("---")
        parts.append("")
    
    # Notable Implementation Details
    if data.get("notable_implementation_details"):
        parts.append("## Notable Implementation Details")
        parts.append("")
        parts.append(data["notable_implementation_details"])
        parts.append("")
    
    return "\n".join(parts)

# Tier order from highest to lowest
TIER_ORDER = ["🥇 Medal", "🔵 Very High", "🟢 High", "🟡 Medium", "🔴 Low", "⚫ Lowest"]

# Tier reference table for prompts/documentation
TIER_REFERENCE = """| Tier | Label | Criteria |
|------|-------|----------|
| 🥇 | **Medal** | Medal-winning solutions |
| 🔵 | **Very High** | percentile ≥ 0.9, no medal |
| 🟢 | **High** | 0.7 ≤ percentile < 0.9 |
| 🟡 | **Medium** | 0.4 ≤ percentile < 0.7 |
| 🔴 | **Low** | 0.2 ≤ percentile < 0.4 |
| ⚫ | **Lowest** | percentile < 0.2 |"""


def assign_tier(rollout_info: dict) -> str:
    """Assign performance tier based on score_percentile and medal_earned.
    
    Tier criteria:
        🥇 Medal: Medal-winning solutions
        🔵 Very High: percentile >= 0.9, no medal
        🟢 High: 0.7 <= percentile < 0.9
        🟡 Medium: 0.4 <= percentile < 0.7
        🔴 Low: 0.2 <= percentile < 0.4
        ⚫ Lowest: percentile < 0.2
    
    Args:
        rollout_info: A dict with 'score_percentile' and 'medal_earned' keys
        
    Returns:
        Tier label string (e.g., "🥇 Medal", "🔵 Very High", etc.)
    """
    medal = rollout_info.get("medal_earned")
    percentile = rollout_info.get("score_percentile")
    
    if medal:  # Medal-winning solution
        return "🥇 Medal"
    
    if percentile is None:
        return "⚫ Lowest"  # Default if no percentile
    
    if percentile >= 0.9:
        return "🔵 Very High"
    elif percentile >= 0.7:
        return "🟢 High"
    elif percentile >= 0.4:
        return "🟡 Medium"
    elif percentile >= 0.2:
        return "🔴 Low"
    else:
        return "⚫ Lowest"




def build_tier_wise_comparison_df(summaries_df: pd.DataFrame) -> list[pd.DataFrame]:
    """Extract pairwise rollouts between adjacent tiers as a list of DataFrames.
    
    Pairs adjacent tiers using a sliding window of size 2, only considering
    tiers that have at least one rollout. Each pair is returned as a DataFrame
    containing rollouts from both tiers.
    
    Args:
        summaries_df: DataFrame with a 'tier' column (assigned by assign_tier)
        
    Returns:
        List of DataFrames, one per adjacent tier pair. Each DataFrame contains
        rollouts from both tiers in the pair, preserving the 'tier' column.
        
    Example:
        If only Medal, High, and Low tiers are populated:
        - pair_1: DataFrame with Medal + High rollouts
        - pair_2: DataFrame with High + Low rollouts
    """
    # Get tiers that have at least one rollout, in order from highest to lowest
    existing_tiers = [tier for tier in TIER_ORDER if tier in summaries_df["tier"].values]
    
    if len(existing_tiers) < 2:
        return []
    
    # Create pairwise comparisons using sliding window of size 2
    pairs = []
    for i in range(len(existing_tiers) - 1):
        tier_high = existing_tiers[i]
        tier_low = existing_tiers[i + 1]
        
        pair_df = summaries_df[summaries_df["tier"].isin([tier_high, tier_low])].copy()
        pairs.append(pair_df)
    
    return pairs


def build_summaries_df(summarization_folder: Path, task_name: str | None = None) -> pd.DataFrame:
    """Build SUMMARIES DataFrame from .md files in the summarization folder.
    
    The .md files are already in markdown format (output from gpt5_mle_summarization_v6.py).
    This function parses the header line to extract metadata and uses the full content as-is.
    
    Args:
        summarization_folder: Path to folder containing .md summary files
        task_name: Optional task name to filter summaries by
        
    Returns:
        DataFrame with columns: rollout_id, score_percentile, medal_earned, task_name,
        tier, full_summary
    """
    import re
    
    md_files = list(summarization_folder.glob("*.md"))
    print(f"Parsing {len(md_files)} .md files...")
    
    summaries_data = []
    for md_file in md_files:
        try:
            with open(md_file) as f:
                content = f.read().strip()
            
            # Skip error files
            if content.startswith("Error:"):
                print(f"Warning: Skipping error file {md_file.name}")
                continue
            
            # Parse header line to extract metadata
            # Format: # Task: {task_name} | Rollout {rollout_id} : Percentile: {percentile} | Tier: {tier} | Medal: {medal}
            header_match = re.match(
                r"^# Task: .+? \| Rollout (\d+) : Percentile: ([\d.]+) \| Tier: ([^|]+) \| Medal: (.+?)$",
                content.split("\n")[0]
            )
            
            if header_match:
                rollout_id = int(header_match.group(1))
                score_percentile = float(header_match.group(2))
                tier = header_match.group(3).strip()
                medal_str = header_match.group(4).strip()
                medal_earned = None if medal_str in ("None", "null") else medal_str
            else:
                print(f"Warning: Could not parse header in {md_file.name}, skipping")
                continue
            
            row = {
                "task_name": task_name,
                "rollout_id": rollout_id,
                "score_percentile": score_percentile,
                "medal_earned": medal_earned,
                "tier": tier,
                "full_summary": content,
            }
            summaries_data.append(row)
        except Exception as e:
            print(f"Warning: Error processing {md_file.name}: {e}")
    
    df = pd.DataFrame(summaries_data)
    print(f"Built SUMMARIES DataFrame with {len(df)} rows")
    return df


def assemble_comparison_report(
    pair_results: list[tuple[list[str], str]],
    task_name: str,
) -> str:
    """Assemble multiple tier pair comparison results into a single markdown report.
    
    Args:
        pair_results: List of tuples (tiers_in_pair, result_response) for each pair
        task_name: Task name for the report header
        
    Returns:
        Formatted markdown string combining all tier pair comparisons
    """
    parts = []
    
    # Report header
    parts.append(f"# Tier-wise Contrastive Analysis: {task_name}")
    parts.append("")
    parts.append(f"**Task:** {task_name}")
    parts.append(f"**Total Tier Pairs Analyzed:** {len(pair_results)}")
    parts.append("")
    parts.append("=" * 80)
    parts.append("")
    
    # Table of contents
    parts.append("## Table of Contents")
    parts.append("")
    for i, (tiers_in_pair, _) in enumerate(pair_results):
        tier_names = " vs ".join(tiers_in_pair)
        parts.append(f"{i + 1}. [{tier_names}](#pair-{i + 1}-{'-vs-'.join([t.split()[1].lower() for t in tiers_in_pair])})")
    parts.append("")
    parts.append("=" * 80)
    parts.append("")
    
    # Each pair's results
    for i, (tiers_in_pair, result_response) in enumerate(pair_results):
        tier_names = " vs ".join(tiers_in_pair)
        anchor_id = f"pair-{i + 1}-{'-vs-'.join([t.split()[1].lower() for t in tiers_in_pair])}"
        
        # Section header with visual separator
        parts.append(f"<a id=\"{anchor_id}\"></a>")
        parts.append("")
        parts.append("=" * 80)
        parts.append(f"## Pair {i + 1}: {tier_names}")
        parts.append("=" * 80)
        parts.append("")
        
        # The actual comparison result
        parts.append(result_response)
        parts.append("")
        parts.append("")
    
    return "\n".join(parts)


def assemble_and_save_combined_report(
    log_dir: str | Path,
    model: str,
    job_name: str,
    run_id: int,
    task_name: str,
) -> Path | None:
    """Read individual pair .md files and assemble into combined report.
    
    Args:
        log_dir: Directory containing pair .md files
        model: Model name used in filenames
        job_name: Job name used in filenames
        run_id: Run ID used in filenames
        task_name: Task name used in filenames
        
    Returns:
        Path to the combined report, or None if no pair files found
    """
    log_dir = Path(log_dir)
    
    print(f"\n{'=' * 80}")
    print("Assembling combined comparison report from individual pair files...")
    print(f"{'=' * 80}")
    
    # Read individual pair .md files and reconstruct pair_results
    pair_results: list[tuple[list[str], str]] = []
    pair_md_files = sorted(
        log_dir.glob(f"{model}_{job_name}_{run_id}_{task_name}_pair*_*.md"),
        key=lambda p: int(p.stem.split("_pair")[1].split("_")[0])  # Sort by pair index
    )
    
    for pair_md_file in pair_md_files:
        with open(pair_md_file) as f:
            content = f.read()
        
        # Extract tier metadata from first line
        first_line = content.split("\n")[0]
        if first_line.startswith("<!-- TIER_METADATA:"):
            metadata_json = first_line.replace("<!-- TIER_METADATA:", "").replace("-->", "").strip()
            metadata = json.loads(metadata_json)
            tiers_in_pair = metadata["tiers"]
            # Remove metadata line from content
            response_content = "\n".join(content.split("\n")[2:])  # Skip metadata + blank line
        else:
            # Fallback: no metadata, use empty tiers
            tiers_in_pair = []
            response_content = content
        
        pair_results.append((tiers_in_pair, response_content))
        print(f"  Loaded: {pair_md_file.name}")
    
    if not pair_results:
        print("ERROR: No pair .md files found to assemble.")
        return None
    
    combined_report = assemble_comparison_report(pair_results, task_name)
    
    # Save combined report with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_path = log_dir / f"{model}_{job_name}_{run_id}_{task_name}_comparison_report_{timestamp}.md"
    with open(report_path, "w") as f:
        f.write(combined_report)
    print(f"\nCombined report saved to: {report_path}")
    
    return report_path
