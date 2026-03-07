"""
Wrapper script to run MLE Bench diversity analysis for multiple tasks.
"""

import subprocess
import sys


def run_for_tasks(summarization_file_names: list[str]) -> None:
    """Run the diversity analysis script for each task name."""
    script_path = "/home/winnieyangwn/rlm/experiments/post-improvement/gpt5_mle_diversity.py"
    
    for i, summarization_file_name in enumerate(summarization_file_names):
        # Build summarization file path based on task name
        summarization_file = f"/checkpoint/maui_sft/winnieyangwn/rlm_dumps/post-improvement/{summarization_file_name}"
        model_name = summarization_file_name.split("_")[0]
        job_name = "post-improvement-diversity"
        run_id = summarization_file_name.split("_")[3]
        task_name = summarization_file_name.split("_")[4]
        print(f"\n{'='*80}")
        print(f"{'='*80}\n")
        try:
            subprocess.run(
                [sys.executable, script_path,
                  "--summarization_file", summarization_file,
                  "--log_dir", "/checkpoint/maui_sft/winnieyangwn/rlm_dumps/post-improvement/",
                  "--run_id", run_id,
                  "--model", model_name,
                  "--job_name", job_name,
                  "--task_name", task_name],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Error processing task '{summarization_file_name}': {e}")
            continue
    print(f"\nCompleted diversity analysis for all {len(summarization_file_names)} tasks.")


if __name__ == "__main__":
    summarization_file_names = [
        "gpt-5_post-improvement_summarization_520-rlm-comparison-argparse-xray_vinbigdata-chest-xray-abnormalities-detection_2026-02-11_01-16-17_2e0b82b1_extracted.jsonl",
        #"iwildcam-2019-fgvc6",
        # 'freesound-audio-tagging-2019',
        # 'dogs-vs-cats-redux-kernels-edition', 
        # 'plant-pathology-2021-fgvc8', 
        # 'plant-pathology-2020-fgvc7', 
        # 'rsna-miccai-brain-tumor-radiogenomic-classification', 
        # 'tabular-playground-series-dec-2021',
    ]
    run_for_tasks(summarization_file_names)
