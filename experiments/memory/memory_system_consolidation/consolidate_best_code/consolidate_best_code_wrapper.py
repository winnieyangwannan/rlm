"""
Wrapper script to extract best code solutions for multiple tasks.

This script runs consolidate_best_code.py for each task in parallel,
extracting the code from the rollout with the highest score.
"""

import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from filelock import FileLock, Timeout


def consolidate_single_task(
    task_name: str,
    script_path: str,
    output_dir: Path,
    run_id: str,
    model: str = "gpt5",
    job_name: str = "code",
    group_name: str = "maui_sft",
    data_paths: list[str] | None = None,
    do_not_skip: list[str] | None = None,
) -> tuple[str, bool, str, float]:
    """Run best code extraction for a single task.
    
    Returns:
        Tuple of (task_name, success, message, elapsed_seconds)
    """
    lock_file = output_dir / f".{task_name}.lock"
    
    try:
        # Acquire lock with 0 timeout - fail immediately if another process holds it
        with FileLock(lock_file, timeout=0):
            # Check if output already exists (skip unless task is in do_not_skip)
            existing_md = list(output_dir.glob(f"*{task_name}*_best.md"))
            if existing_md:
                if task_name not in (do_not_skip or []):
                    return (task_name, True, f"Skipped (exists): {existing_md[0].name}", 0.0)
                # Delete existing file(s) if task is in do_not_skip
                for md_file in existing_md:
                    md_file.unlink()
            
            start_time = time.time()
            env = os.environ.copy()
            env["PYTHONPATH"] = "/home/winnieyangwn/rlm:" + env.get("PYTHONPATH", "")
            
            cmd = [
                sys.executable,
                script_path,
                "--task_name", task_name,
                "--output_dir", str(output_dir),
                "--run_id", run_id,
                "--model", model,
                "--job_name", job_name,
                "--group_name", group_name,
            ]
            if data_paths:
                cmd.extend(["--data_paths"] + data_paths)
            
            subprocess.run(
                cmd,
                check=True,
                cwd="/home/winnieyangwn/rlm",
                env=env,
            )
            elapsed = time.time() - start_time
            return (task_name, True, "Completed", elapsed)
    except Timeout:
        return (task_name, True, "Skipped (another process is handling this task)", 0.0)
    except subprocess.CalledProcessError as e:
        return (task_name, False, f"Error: exit code {e.returncode}", 0.0)


def consolidate_for_tasks(
    task_names: list[str],
    output_dir: str,
    script_path: str,
    run_id: str,
    max_workers: int = 4,
    model: str = "gpt5",
    job_name: str = "code",
    group_name: str = "maui_sft",
    data_paths: list[str] | None = None,
    do_not_skip: list[str] | None = None,
) -> None:
    """Run the best code extraction script for each task name in parallel.
    
    Args:
        task_names: List of task names to process.
        output_dir: Directory to save output .md files.
        script_path: Path to the best code extraction script.
        run_id: Run ID for the analysis.
        max_workers: Number of parallel workers (default: 4).
        model: Model name for file naming (default: gpt5).
        job_name: Job name for file naming (default: code).
        group_name: Checkpoint group name (default: maui_sft).
        data_paths: List of paths to episodic memory JSONL files.
        do_not_skip: List of task names to always process even if output exists.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Cap workers to task count to avoid idle workers
    max_workers = min(max_workers, len(task_names))
    print(f"Extracting best code for {len(task_names)} tasks with {max_workers} workers...")
    
    results = {"success": [], "failed": [], "skipped": []}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                consolidate_single_task, task, script_path, output_path, run_id,
                model, job_name, group_name, data_paths, do_not_skip
            ): task 
            for task in task_names
        }
        
        for future in as_completed(futures):
            task_name, success, message, elapsed = future.result()
            time_str = f" ({elapsed:.1f}s)" if elapsed > 0 else ""
            if "Skipped" in message:
                results["skipped"].append(task_name)
                reason = message.replace("Skipped ", "").replace("Skipped", "").strip()
                print(f"⏭️  {task_name}: Skipped. Reason: {reason}")
            elif success:
                results["success"].append(task_name)
                print(f"✅ {task_name}: {message}{time_str}")
            else:
                results["failed"].append(task_name)
                print(f"❌ {task_name}: {message}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Summary: {len(results['success'])} completed, {len(results['skipped'])} skipped, {len(results['failed'])} failed")
    if results["failed"]:
        print(f"Failed tasks: {results['failed']}")
    print(f"\nBest code files saved to: {output_dir}")
