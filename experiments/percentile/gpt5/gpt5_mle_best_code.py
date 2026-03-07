"""
Script to extract code solution from rollout with highest score.

This script:
1. Loads data from a JSONL file as pandas DataFrame
2. Finds the rollout with the highest score
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
    return parser.parse_args()


def get_data_path(run_id: str, group_name: str = "maui_sft") -> str:
    """Construct data path from run_id."""
    return f"/checkpoint/{group_name}/winnieyangwn/amaia_dumps/{run_id}/trajectories/{run_id}_metadata.jsonl"


def load_data(data_path: str) -> pd.DataFrame:
    """Load JSONL data as pandas DataFrame."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_json(data_path, lines=True)
    print(f"Loaded {len(df)} rows from {data_path}")
    return df


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
    
    # Filter for rows with valid scores
    filtered_df = filtered_df[filtered_df[score_col].notna()]
    
    if len(filtered_df) == 0:
        print("No rollouts with valid scores found.")
        return filtered_df
    
    # Find the row with the highest score
    best_idx = filtered_df[score_col].idxmax()
    best_df = filtered_df.loc[[best_idx]]
    
    best_score = best_df[score_col].iloc[0]
    print(f"Found best rollout with {score_col}: {best_score}")
    
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
        print(f"Saved: {file_name} (score: {score})")
    
    return saved_files


def main():
    args = parse_args()
    
    # 1. Construct data path and load as pandas df
    data_path = get_data_path(args.run_id, args.group_name)
    df = load_data(data_path)
    
    # 2. Find rollout with highest score
    best_df = find_best_rollout(df, args.task_name)
    
    if len(best_df) == 0:
        print("No rollouts with valid scores found.")
        return
    
    # 3. Extract code solution and save as .md file
    saved_files = extract_and_save_code(
        df=best_df,
        output_dir=args.output_dir,
        model=args.model,
        job_name=args.job_name,
        run_id=args.run_id,
    )
    
    print(f"\nSaved {len(saved_files)} code files to {args.output_dir}")


if __name__ == "__main__":
    main()
