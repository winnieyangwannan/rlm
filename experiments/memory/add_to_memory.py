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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import data.data_utils as data_utils
from data.data_utils import *


def main(run_id: str, group: str, 
         only_valid_submissions: bool = False, 
         memory_path: str | None = None, generation_id: str | None = None,
         trajectories_dir: str | None = None):
    """
    Process MLE Bench trajectory data and save flattened metadata.
    
    Args:
        run_id: The run identifier (e.g., '503')
        group: The checkpoint group (e.g., 'agentic-models')
        only_valid_submissions: If True, only keep rollouts with valid_submission=True.
            Defaults to False.
        memory_path: Optional path to save df_flat in a folder named '{run_id}'.
        generation_id: Optional generation identifier used in the output filename.
        trajectories_dir: Optional path to the trajectories folder. Defaults to
            '/checkpoint/{group}/winnieyangwn/amaia_dumps/{run_id}/trajectories'.
    """
    # Dynamically find the jsonl file in the trajectories folder
    if trajectories_dir is None:
        trajectories_dir = f'/checkpoint/{group}/winnieyangwn/amaia_dumps/{run_id}/trajectories'
    
    print(f"\n{'='*60}")
    print(f"ADD TO MEMORY - Converting Trajectories to Raw Memories")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  run_id: {run_id}")
    print(f"  group: {group}")
    print(f"  generation_id: {generation_id}")
    print(f"  only_valid_submissions: {only_valid_submissions}")
    print(f"{'='*60}")
    print(f"Paths:")
    print(f"  Source (trajectories): {trajectories_dir}")
    print(f"  Destination (memory):  {memory_path}")
    print(f"{'='*60}\n")
    
    # Find subdirectory containing the jsonl file
    subdirs = [d for d in os.listdir(trajectories_dir) 
               if os.path.isdir(os.path.join(trajectories_dir, d)) and d.startswith('mle_bench_')]
    
    if not subdirs:
        raise FileNotFoundError(f"No mle_bench_* subdirectory found in {trajectories_dir}")
    
    subdir = subdirs[0]  # Use the first matching subdirectory
    subdir_path = os.path.join(trajectories_dir, subdir)
    print(f"Found subdirectory: {subdir}")
    
    # Find the jsonl file in the subdirectory
    jsonl_files = [f for f in os.listdir(subdir_path) if f.endswith('.jsonl')]
    
    if not jsonl_files:
        raise FileNotFoundError(f"No .jsonl file found in {subdir_path}")
    
    file_path = os.path.join(subdir_path, jsonl_files[0])
    print(f"Loading trajectory data from: {file_path}")
    mle_bench_data_dir = "/checkpoint/maui_sft/winnieyangwn/datasets"
    # Load trajectory data
    df_trj = pd.read_json(file_path, lines=True)
    print(f"Loaded {len(df_trj)} trajectories from file")
    
    # Define error categories to filter out (infrastructure errors)
    filter_out_errors = [
        "Too many streaming blocks of output",
        "Infrastructure: Connection/Heartbeat Error",
        "Infrastructure: AgentBox Backend Not Initialized",
        "Infrastructure: Container Error",
        "Infrastructure: Container Execution Error",
        "Infrastructure: Container Creation Error",
        "Infrastructure: Model Call Error",
    ]
    
    # Process trajectories as episodic memories
    print(f"\nProcessing trajectories as episodic memories...")
    print(f"Filtering out infrastructure errors:")
    for err in filter_out_errors:
        print(f"  - {err}")
    if only_valid_submissions:
        print(f"Filtering: Only keeping valid submissions")
    
    df_flat, infra_error_indices = flatten_dataframe(
        df_trj, 
        only_valid_submissions=only_valid_submissions,
        mle_bench_data_dir=mle_bench_data_dir,
        filter_out_errors=filter_out_errors,
        return_infra_error_indices=True
    )
    
    # Report filtering results
    num_filtered = len(df_trj) - len(df_flat)
    print(f"\nFiltering Results:")
    print(f"  Original trajectories: {len(df_trj)}")
    print(f"  Infrastructure errors found: {len(infra_error_indices)}")
    print(f"  Total filtered out: {num_filtered}")
    print(f"  Remaining memories: {len(df_flat)}")
    
    # Report task distribution
    if len(df_flat) > 0:
        unique_tasks = df_flat['task_name'].nunique()
        valid_count = df_flat['valid_submission'].sum() if 'valid_submission' in df_flat.columns else 0
        print(f"\nMemory Statistics:")
        print(f"  Unique tasks: {unique_tasks}")
        print(f"  Valid submissions: {valid_count}")
        print(f"  Invalid submissions: {len(df_flat) - valid_count}")
    
    print(f"\nEpisodic Memory DataFrame:")
    print(f"  Shape: {df_flat.shape}")

    output_name = f'episodic_memory_{generation_id}.jsonl'
    

    # Save episodic memories to memory folder
    if memory_path is not None:
        memory_folder = os.path.join(memory_path, f'{generation_id}')
        os.makedirs(memory_folder, exist_ok=True)
        memory_save_path = os.path.join(memory_folder, output_name)
        
        print(f"\nSaving raw memories...")
        print(f"  Output folder: {memory_folder}")
        print(f"  Output file: {output_name}")
        
        # Check if file already exists - if so, append instead of overwriting
        if os.path.exists(memory_save_path):
            existing_df = pd.read_json(memory_save_path, lines=True)
            existing_count = len(existing_df)
            print(f"  Found existing memory file with {existing_count} records - APPENDING (not overwriting)")
            combined_df = pd.concat([existing_df, df_flat], ignore_index=True)
            combined_df.to_json(memory_save_path, orient='records', lines=True)
            new_total = len(combined_df)
            print(f"\n{'='*60}")
            print(f"EPISODIC MEMORIES APPENDED SUCCESSFULLY")
            print(f"{'='*60}")
            print(f"  Full path: {memory_save_path}")
            print(f"  Existing memories: {existing_count}")
            print(f"  New memories added: {len(df_flat)}")
            print(f"  Total memories now: {new_total}")
            print(f"{'='*60}\n")
        else:
            df_flat.to_json(memory_save_path, orient='records', lines=True)
            print(f"\n{'='*60}")
            print(f"EPISODIC MEMORIES SAVED SUCCESSFULLY")
            print(f"{'='*60}")
            print(f"  Full path: {memory_save_path}")
            print(f"  Total raw memories: {len(df_flat)}")
            print(f"{'='*60}\n")
    else:
        print(f"\nWarning: memory_path not provided. Raw memories not saved.")
    
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
    parser.add_argument("--memory_path", type=str, default="/checkpoint/agentic-models/winnieyangwn/memory", help="Path to save df_flat in a folder named '{run_id}'")
    parser.add_argument("--generation_id", type=str, default=None, help="Generation identifier used in the output filename")
    parser.add_argument("--trajectories_dir", type=str, default=None, help="Path to trajectories folder (default: /checkpoint/{group}/winnieyangwn/amaia_dumps/{run_id}/trajectories)")

    args = parser.parse_args()
    main(run_id=args.run_id, group=args.group,
          only_valid_submissions=args.only_valid_submissions,
          memory_path=args.memory_path, generation_id=args.generation_id,
          trajectories_dir=args.trajectories_dir)