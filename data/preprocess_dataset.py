import pandas as pd
import os
import gzip
import json
import zlib
import numpy as np
import sys
import ast
import importlib
import argparse

# Add the rlm project root to path so data.data_utils can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data.data_utils as data_utils
from data.data_utils import *

# python /home/winnieyangwn/rlm/data/preprocess_dataset.py --run_id 524_code_mle_30_r2_7 --group agentic-models --only_valid_submissions False --code_only False

def main(run_id: str, group: str, 
         only_valid_submissions: bool = False, 
         code_only: bool = False,
         output_dir: str | None = None, output_name: str | None = None):
    """
    Process MLE Bench trajectory data and save flattened metadata.
    
    Args:
        run_id: The run identifier (e.g., '503')
        group: The checkpoint group (e.g., 'agentic-models')
        only_valid_submissions: If True, only keep rollouts with valid_submission=True.
            Defaults to False.
        output_dir: Optional output directory. Defaults to the trajectories folder.
        output_name: Optional output filename. Defaults to '{run_id}_metadata.jsonl'.
    """
    # Dynamically find the jsonl file in the trajectories folder
    trajectories_dir = f'/checkpoint/{group}/winnieyangwn/amaia_dumps/{run_id}/trajectories'
    
    # Find subdirectory containing the jsonl file
    subdirs = [d for d in os.listdir(trajectories_dir) 
               if os.path.isdir(os.path.join(trajectories_dir, d)) and d.startswith('mle_bench_')]
    
    if not subdirs:
        raise FileNotFoundError(f"No mle_bench_* subdirectory found in {trajectories_dir}")
    
    subdir = subdirs[0]  # Use the first matching subdirectory
    subdir_path = os.path.join(trajectories_dir, subdir)
    
    # Find the jsonl file in the subdirectory
    jsonl_files = [f for f in os.listdir(subdir_path) if f.endswith('.jsonl')]
    
    if not jsonl_files:
        raise FileNotFoundError(f"No .jsonl file found in {subdir_path}")
    
    file_path = os.path.join(subdir_path, jsonl_files[0])
    print(f"Loading trajectory data from: {file_path}")
    mle_bench_data_dir = "/checkpoint/maui_sft/winnieyangwn/datasets"
    # Load trajectory data
    df_trj = pd.read_json(file_path, lines=True)
    
    # Flatten the dataframe
    df_flat = flatten_dataframe(df_trj, only_valid_submissions=only_valid_submissions,
                                mle_bench_data_dir=mle_bench_data_dir, code_only=code_only)
    
    print(f"Shape: {df_flat.shape}")
    print(f"Columns: {df_flat.columns.tolist()}")
    
    # Save flattened data
    if output_dir is None:
        output_dir = trajectories_dir
    if output_name is None:
        if not code_only:
            output_name = f'{run_id}_metadata.jsonl'
        else:
            output_name = f'{run_id}_metadata_code_only.jsonl'
    
    save_path = os.path.join(output_dir, output_name)
    df_flat.to_json(save_path, orient='records', lines=True)
    print(f"Saved {len(df_flat)} rows to {save_path}")
    
    return df_flat


def str_to_bool(v: str) -> bool:
    """Convert string to boolean for argparse."""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected, got '{v}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess MLE Bench trajectory data")
    parser.add_argument("--run_id", type=str, default="514", help="Run identifier (e.g., '503')")
    parser.add_argument("--group", type=str, default="maui_sft", help="Checkpoint group (e.g., 'agentic-models')")
    parser.add_argument("--only_valid_submissions", type=str_to_bool, default=False, help="Only keep rollouts with valid submissions")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: trajectories folder)")
    parser.add_argument("--output_name", type=str, default=None, help="Output filename (default: '{run_id}_metadata.jsonl')")
    parser.add_argument("--code_only", type=str_to_bool, default=False, help="Only include code-related columns in the output")

    args = parser.parse_args()
    main(run_id=args.run_id, group=args.group,
          only_valid_submissions=args.only_valid_submissions,
          code_only=args.code_only,
          output_dir=args.output_dir, output_name=args.output_name)