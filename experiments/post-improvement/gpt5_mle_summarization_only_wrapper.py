"""
Wrapper script to run MLE Bench summarization analysis for multiple tasks.
"""

import subprocess
import sys


def run_for_tasks(task_names: list[str]) -> None:
    """Run the summarization script for each task name."""
    script_path = "/home/winnieyangwn/rlm/experiments/post-improvement/gpt5_mle_summarization_only.py"
    
    print(f"Running analysis for {len(task_names)} tasks...")
    for i, task_name in enumerate(task_names):
        print(f"\n{'='*80}")
        print(f"Task {i+1}/{len(task_names)}: {task_name}")
        print(f"{'='*80}\n")
        try:
            subprocess.run(
                [sys.executable, script_path,
                  "--account", "agentic-models",#"maui_sft",
                  "--run-id", "520-rlm-comparison-argparse-xray",
                  "--task-name", task_name,
                  "--log-dir", "/checkpoint/maui_sft/winnieyangwn/rlm_dumps/post-improvement/"],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Error processing task '{task_name}': {e}")
            continue
    print(f"\nCompleted analysis for all {len(task_names)} tasks.")


if __name__ == "__main__":
    task_names = [
        "vinbigdata-chest-xray-abnormalities-detection",
        #"iwildcam-2019-fgvc6",
        # 'freesound-audio-tagging-2019',
        # 'dogs-vs-cats-redux-kernels-edition', 
        # 'plant-pathology-2021-fgvc8', 
        # 'plant-pathology-2020-fgvc7', 
        # 'rsna-miccai-brain-tumor-radiogenomic-classification', 
        # 'tabular-playground-series-dec-2021',
    ]
    run_for_tasks(task_names)
