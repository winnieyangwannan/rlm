"""
Script to extract code solution from rollout with highest score.

This script:
1. Loads data from multiple JSONL files as pandas DataFrame
2. Finds the rollout with the highest score across all sources
3. Extracts the code solution and saves as .md file
4. Names files as: {model}_{job_name}_{run_id}_{task_name}_rollout{row_id}_best.md
"""

import argparse
import os
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Extract code from medal rollouts")
    parser.add_argument(
        "--run_id",
        type=str,
        default="514",
        help="Run ID (used to construct data path and for file naming)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/checkpoint/maui_sft/winnieyangwn/rlm_dumps/best_code/mle_30",
        help="Directory to save output .md files",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt5",
        help="Model name for file naming",
    )
    parser.add_argument(
        "--job_name",
        type=str,
        default="code",
        help="Job name for file naming",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="freesound-audio-tagging-2019",
        help="Filter by specific task name (optional)",
    )
    parser.add_argument(
        "--group_name",
        type=str,
        default="maui_sft",
        help="Group name for checkpoint path",
    )
    parser.add_argument(
        "--data_paths",
        type=str,
        nargs="*",
        default=[],
        help="List of paths to episodic memory JSONL files (overrides constructed path)",
    )
    return parser.parse_args()


def get_data_path(run_id: str, group_name: str = "maui_sft") -> str:
    """Construct data path from run_id."""
    return f"/checkpoint/{group_name}/winnieyangwn/amaia_dumps/{run_id}/trajectories/{run_id}_metadata.jsonl"


def load_data(data_paths: list[str]) -> pd.DataFrame:
    """Load JSONL data from multiple files and combine as pandas DataFrame.
    
    Args:
        data_paths: List of paths to JSONL files
        
    Returns:
        Combined DataFrame with all rollouts and a 'source_file' column
    """
    all_dfs = []
    
    for data_path in data_paths:
        if not os.path.exists(data_path):
            print(f"  ⚠️  Skipping missing file: {data_path}")
            continue
        
        df = pd.read_json(data_path, lines=True)
        df["source_file"] = data_path  # Track which file each row came from
        unique_tasks = df['task_name'].nunique() if 'task_name' in df.columns else 0
        print(f"  ✓ Loaded {len(df)} rows ({unique_tasks} unique tasks) from {Path(data_path).name}")
        all_dfs.append(df)
    
    if not all_dfs:
        raise FileNotFoundError(f"No valid data files found in: {data_paths}")
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    total_tasks = combined_df['task_name'].nunique() if 'task_name' in combined_df.columns else 0
    print(f"  ✓ Combined: {len(combined_df)} total rows ({total_tasks} unique tasks) from {len(all_dfs)} files")
    
    return combined_df


def find_best_rollout(df: pd.DataFrame, task_name: str | None = None, score_col: str = "percentile") -> pd.DataFrame:
    """Find the rollout with the highest score.
    
    Args:
        df: Input DataFrame
        task_name: Optional filter by task name
        score_col: Column name containing the score (default: "percentile")
        
    Returns:
        DataFrame containing only the row with the highest score
    """
    filtered_df = df.copy()
    
    if task_name:
        filtered_df = filtered_df[filtered_df["task_name"] == task_name]
        print(f"  Filtering for task: '{task_name}'")
        print(f"  Rollouts for this task: {len(filtered_df)}")
    
    # Filter for rows with valid scores
    filtered_df = filtered_df[filtered_df[score_col].notna()]
    
    if len(filtered_df) == 0:
        print(f"  ⚠️  No rollouts with valid {score_col} scores found.")
        return filtered_df
    
    print(f"  Rollouts with valid {score_col}: {len(filtered_df)}")
    
    # Find the row with the highest score
    best_idx = filtered_df[score_col].idxmax()
    best_df = filtered_df.loc[[best_idx]]
    
    best_score = best_df[score_col].iloc[0]
    best_medal = best_df['medal'].iloc[0] if 'medal' in best_df.columns else 'N/A'
    best_source = best_df['source_file'].iloc[0] if 'source_file' in best_df.columns else 'N/A'
    print(f"  ✓ Best rollout: {score_col}={best_score}, medal={best_medal}, idx={best_idx}")
    print(f"    Source: {best_source}")
    
    return best_df


def extract_and_save_code(
    df: pd.DataFrame,
    output_dir: str,
    model: str,
    job_name: str,
    run_id: str,
) -> list[str]:
    """Extract code from medal rollouts and save as .md files.
    
    Args:
        df: DataFrame with medal rollouts
        output_dir: Directory to save output files
        model: Model name for file naming
        job_name: Job name for file naming
        run_id: Run ID for file naming
        
    Returns:
        List of saved file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_files = []
    
    for idx, row in df.iterrows():
        task_name = row.get("task_name", "unknown")
        code = row.get("code", "")
        medal = row.get("medal", "")
        percentile = row.get("percentile", "N/A")
        
        # Use the DataFrame index as the row_id
        row_id = idx
        
        score = row.get("score", "N/A")
        
        # Create filename
        file_name = f"{model}_{job_name}_{run_id}_{task_name}_rollout{row_id}_best.md"
        file_path = os.path.join(output_dir, file_name)
        
        # Create markdown content with metadata header
        content = f"""# Code Solution - {task_name}

## Metadata
- **Model**: {model}
- **Job Name**: {job_name}
- **Run ID**: {run_id}
- **Task Name**: {task_name}
- **Rollout ID**: {row_id}
- **Score**: {score}
- **Percentile**: {percentile}
- **Medal**: {medal}

## Code

```python
{code}
```
"""
        
        # Save to file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        saved_files.append(file_path)
        print(f"  ✓ Saved: {file_name} (percentile: {percentile}, medal: {medal})")
    
    return saved_files


def main():
    args = parse_args()
    
    print(f"\n{'='*60}")
    print(f"BEST CODE EXTRACTION - System Consolidation")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  Task:       {args.task_name}")
    print(f"  Run ID:     {args.run_id}")
    print(f"  Model:      {args.model}")
    print(f"  Job name:   {args.job_name}")
    print(f"  Output dir: {args.output_dir}")
    
    # 1. Get data paths (use provided paths or construct from run_id)
    if args.data_paths:
        data_paths = args.data_paths
        print(f"  Data paths: {len(data_paths)} file(s) provided")
        for i, p in enumerate(data_paths):
            print(f"    [{i+1}] {p}")
    else:
        data_paths = [get_data_path(args.run_id, args.group_name)]
        print(f"  Data path:  {data_paths[0]} (constructed)")
    print(f"{'='*60}\n")
    
    print(f"[Step 1/3] Loading episodic memory data from {len(data_paths)} source(s)...")
    df = load_data(data_paths)
    
    # 2. Find rollout with highest score
    print(f"\n[Step 2/3] Finding best rollout...")
    best_df = find_best_rollout(df, args.task_name)
    
    if len(best_df) == 0:
        print(f"\n{'='*60}")
        print(f"❌ FAILED: No valid rollouts found for task '{args.task_name}'")
        print(f"{'='*60}\n")
        return
    
    # 3. Extract code solution and save as .md file
    print(f"\n[Step 3/3] Extracting and saving best code...")
    saved_files = extract_and_save_code(
        df=best_df,
        output_dir=args.output_dir,
        model=args.model,
        job_name=args.job_name,
        run_id=args.run_id,
    )
    
    print(f"\n{'='*60}")
    print(f"✅ SUCCESS: Saved {len(saved_files)} semantic memory file(s)")
    print(f"   Output: {args.output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
