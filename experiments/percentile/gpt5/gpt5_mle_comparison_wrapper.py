"""
Wrapper script to run MLE Bench comparison analysis for multiple tasks.
"""

import subprocess
import sys


def run_for_tasks(task_names: list[str], summarization_logs: list[str]) -> None:
    """Run the comparison script for each task name and its corresponding summarization log.
    
    Args:
        task_names: List of task names to analyze.
        summarization_logs: List of paths to summarization log files (must match task_names order).
    """
    if len(task_names) != len(summarization_logs):
        raise ValueError(f"task_names ({len(task_names)}) and summarization_logs ({len(summarization_logs)}) must have the same length")
    
    script_path = "/home/winnieyangwn/rlm/experiments/percentile/gpt5/gpt5_mle_comparison.py"
    
    print(f"Running comparison analysis for {len(task_names)} tasks...")
    for i, (task_name, summarization_log) in enumerate(zip(task_names, summarization_logs)):
        print(f"\n{'='*80}")
        print(f"Task {i+1}/{len(task_names)}: {task_name}")
        print(f"Summarization log: {summarization_log}")
        print(f"{'='*80}\n")
        try:
            subprocess.run(
                [
                    sys.executable,
                    script_path,
                    "--task-name", task_name,
                    "--summarization-log", summarization_log,
                ],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Error processing task '{task_name}': {e}")
            continue
    print(f"\nCompleted comparison analysis for all {len(task_names)} tasks.")


if __name__ == "__main__":
    task_names = [
        "iwildcam-2019-fgvc6",
        # "freesound-audio-tagging-2019"
        # Add more task names here
    ]
    summarization_logs = [
        "/checkpoint/maui_sft/winnieyangwn/rlm_dumps/gpt-5_summarization_513_iwildcam-2019-fgvc6_2026-02-09_07-05-37_3bfb9fbc.jsonl",
        # "/checkpoint/maui_sft/winnieyangwn/rlm_dumps/archive/gpt5/summarization/gpt-5_summarization_513_freesound-audio-tagging-2019_2026-02-09_08-07-23_4a18efaa.jsonl"
    ]
    run_for_tasks(task_names, summarization_logs)
