"""
System Consolidation - Convert episodic memories to semantic memories.

This module implements the system consolidation process from cognitive science,
where raw episodic memories (individual trajectories/experiences) are transformed
into consolidated semantic memories (generalized knowledge like best code solutions).

Supported consolidation modes:
- best_code: Extract the highest-scoring code solution for each task

Usage:
    python system_consolidation.py --episodic_memory_path /path/to/episodic.jsonl \
        --consolidation_mode best_code --semantic_memory_dir /path/to/output
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path

# Add the rlm project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from experiments.memory.memory_system_consolidation.consolidate_best_code.consolidate_best_code_wrapper import consolidate_for_tasks


# Available consolidation modes
CONSOLIDATION_MODES = ["best_code"]


def _parse_episodic_memory_path(episodic_memory_path: str) -> dict[str, str]:
    """
    Parse episodic memory path to extract group_name, model, and run_id.
    
    Expected path format:
    /checkpoint/{group_name}/winnieyangwn/memory/{model}/{run_id}/episodic_memory/...
    
    Returns:
        Dict with keys: group_name, model, run_id
    """
    parts = episodic_memory_path.split('/')
    result = {}
    
    # Find 'checkpoint' index and extract group_name (next element)
    if 'checkpoint' in parts:
        checkpoint_idx = parts.index('checkpoint')
        if checkpoint_idx + 1 < len(parts):
            result['group_name'] = parts[checkpoint_idx + 1]
    
    # Find 'memory' index and extract model and run_id (next two elements)
    if 'memory' in parts:
        memory_idx = parts.index('memory')
        if memory_idx + 1 < len(parts):
            result['model'] = parts[memory_idx + 1]
        if memory_idx + 2 < len(parts):
            result['run_id'] = parts[memory_idx + 2]
    
    return result


def system_consolidation(
    episodic_memory_paths: list[str],
    consolidation_mode: str,
    semantic_memory_dir: str | None = None,
    run_id: str | None = None,
    max_workers: int = 4,
    model: str | None = None,
    job_name: str = "code",
    group_name: str | None = None,
    do_not_skip: list[str] | None = None,
) -> None:
    """
    Consolidate episodic memories into semantic memories.
    
    Args:
        episodic_memory_paths: List of paths to episodic memory .jsonl files.
        consolidation_mode: Mode of consolidation. Currently supported: "best_code".
        semantic_memory_dir: Directory to save consolidated semantic memories. 
            Defaults to same directory as first episodic_memory_path.
        run_id: Run ID for naming output files. Defaults to extracting from path.
        max_workers: Number of parallel workers for processing (default: 4).
        model: Model name for file naming. Defaults to extracting from path.
        job_name: Job name for file naming (default: code).
        group_name: Checkpoint group name. Defaults to extracting from path.
        do_not_skip: List of task names to always process even if output exists.
    """
    print(f"\n{'='*60}")
    print(f"SYSTEM CONSOLIDATION - Converting Episodic to Semantic Memory")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  Episodic memory paths: {len(episodic_memory_paths)} file(s)")
    for i, p in enumerate(episodic_memory_paths):
        print(f"    [{i+1}] {p}")
    print(f"  Consolidation mode: {consolidation_mode}")
    print(f"  Semantic memory directory: {semantic_memory_dir}")
    print(f"  Max workers: {max_workers}")
    print(f"{'='*60}\n")
    
    # Validate consolidation mode
    print(f"[Step 1/5] Validating consolidation mode...")
    if consolidation_mode not in CONSOLIDATION_MODES:
        raise ValueError(f"Unknown consolidation_mode: {consolidation_mode}. "
                        f"Supported modes: {CONSOLIDATION_MODES}")
    print(f"  ✓ Consolidation mode '{consolidation_mode}' is valid")
    
    # Validate episodic memory paths exist
    print(f"\n[Step 2/5] Validating episodic memory paths...")
    valid_paths = []
    for path in episodic_memory_paths:
        if os.path.exists(path):
            valid_paths.append(path)
            print(f"  ✓ Found: {path}")
        else:
            print(f"  ⚠️  Not found (skipping): {path}")
    
    if not valid_paths:
        raise FileNotFoundError(f"No valid episodic memory files found in: {episodic_memory_paths}")
    print(f"  ✓ {len(valid_paths)} valid episodic memory file(s)")
    
    # Load episodic memory from all paths to extract task names
    print(f"\n[Step 3/5] Loading episodic memory from {len(valid_paths)} source(s)...")
    all_dfs = []
    for path in valid_paths:
        print(f"  Reading from: {path}")
        df = pd.read_json(path, lines=True)
        print(f"    ✓ Loaded {len(df)} entries")
        all_dfs.append(df)
    
    df_episodic = pd.concat(all_dfs, ignore_index=True)
    print(f"  ✓ Combined: {len(df_episodic)} total episodic memory entries")
    
    # Extract unique task names
    task_names = df_episodic['task_name'].unique().tolist()
    print(f"  ✓ Found {len(task_names)} unique tasks to consolidate")
    
    # Set default semantic memory directory
    print(f"\n[Step 4/5] Setting up output directory...")
    if semantic_memory_dir is None:
        semantic_memory_dir = os.path.dirname(valid_paths[0])
        print(f"  Using default directory (same as first episodic memory path)")
    
    # Append consolidation_mode as subdirectory
    semantic_memory_dir = os.path.join(semantic_memory_dir, consolidation_mode)
    os.makedirs(semantic_memory_dir, exist_ok=True)
    print(f"  Semantic memory directory: {semantic_memory_dir}")
    
    # Extract run_id, model, and group_name from first path if not provided
    parsed = _parse_episodic_memory_path(valid_paths[0])
    
    if run_id is None:
        run_id = parsed.get('run_id', 'unknown')
        print(f"  Extracted run_id from path: {run_id}")
    else:
        print(f"  Using provided run_id: {run_id}")
    
    if model is None:
        model = parsed.get('model', 'gpt5')
        print(f"  Extracted model from path: {model}")
    else:
        print(f"  Using provided model: {model}")
    
    if group_name is None:
        group_name = parsed.get('group_name', 'agentic-models')
        print(f"  Extracted group_name from path: {group_name}")
    else:
        print(f"  Using provided group_name: {group_name}")
    
    # Dispatch to appropriate consolidation function
    print(f"\n[Step 5/5] Running consolidation...")
    if consolidation_mode == "best_code":
        print(f"  Mode: best_code - Extracting highest-scoring code for each task")
        _consolidate_best_code(
            task_names=task_names,
            semantic_memory_dir=semantic_memory_dir,
            run_id=run_id,
            max_workers=max_workers,
            model=model,
            job_name=job_name,
            group_name=group_name,
            episodic_memory_paths=valid_paths,
            do_not_skip=do_not_skip,
        )
    
    print(f"\n{'='*60}")
    print(f"SYSTEM CONSOLIDATION COMPLETE")
    print(f"{'='*60}\n")


def _consolidate_best_code(
    task_names: list[str],
    semantic_memory_dir: str,
    run_id: str,
    max_workers: int = 4,
    model: str = "gpt5",
    job_name: str = "code",
    group_name: str = "agentic-models",
    episodic_memory_paths: list[str] | None = None,
    do_not_skip: list[str] | None = None,
) -> None:
    """
    Consolidate episodic memories by extracting best code for each task.
    
    Uses best_code_wrapper to extract the code from the rollout with the
    highest score for each task.
    """
    # Path to the best code extraction script
    script_path = os.path.join(
        os.path.dirname(__file__), 
        "consolidate_best_code",
        "consolidate_best_code.py"
    )
    
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Best code extraction script not found: {script_path}")
    
    print(f"  Script path: {script_path}")
    print(f"  Output directory: {semantic_memory_dir}")
    print(f"  Tasks to process: {len(task_names)}")
    print(f"  Parallel workers: {max_workers}")
    print(f"\n  Starting parallel consolidation...")
    
    consolidate_for_tasks(
        task_names=task_names,
        output_dir=semantic_memory_dir,
        script_path=script_path,
        run_id=run_id,
        max_workers=max_workers,
        model=model,
        job_name=job_name,
        group_name=group_name,
        data_paths=episodic_memory_paths,
        do_not_skip=do_not_skip,
    )


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="System Consolidation - Convert episodic memories to semantic memories"
    )
    parser.add_argument(
        "--episodic_memory_paths", 
        type=str, 
        nargs="+",
        required=True,
        help="Path(s) to the episodic memory .jsonl file(s)"
    )
    parser.add_argument(
        "--consolidation_mode", 
        type=str, 
        default="best_code",
        choices=CONSOLIDATION_MODES,
        help="Consolidation mode (default: best_code)"
    )
    parser.add_argument(
        "--semantic_memory_dir", 
        type=str, 
        default=None,
        help="Directory to save consolidated semantic memories"
    )
    parser.add_argument(
        "--run_id", 
        type=str, 
        default=None,
        help="Run ID for naming output files (extracted from path if not provided)"
    )
    parser.add_argument(
        "--max_workers", 
        type=int, 
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default=None,
        help="Model name for file naming (extracted from path if not provided)"
    )
    parser.add_argument(
        "--job_name", 
        type=str, 
        default="code",
        help="Job name for file naming (default: code)"
    )
    parser.add_argument(
        "--group_name", 
        type=str, 
        default=None,
        help="Checkpoint group name (extracted from path if not provided)"
    )
    parser.add_argument(
        "--do_not_skip", 
        type=str, 
        nargs="+",
        default=None,
        help="Task name(s) to always process even if output already exists"
    )
    
    args = parser.parse_args()
    
    system_consolidation(
        episodic_memory_paths=args.episodic_memory_paths,
        consolidation_mode=args.consolidation_mode,
        semantic_memory_dir=args.semantic_memory_dir,
        run_id=args.run_id,
        max_workers=args.max_workers,
        model=args.model,
        job_name=args.job_name,
        group_name=args.group_name,
        do_not_skip=args.do_not_skip,
    )


if __name__ == "__main__":
    main()
