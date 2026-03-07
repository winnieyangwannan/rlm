"""
Wrapper script to run MLE Bench comparison analysis for multiple tasks.
"""

import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from filelock import FileLock, Timeout


def has_regeneration_error(md_content: str) -> bool:
    """Check if the md content contains error messages that require regeneration."""
    # Check for timeout error
    if "Error: Request failed: timed out" in md_content:
        return True
    # Check for "Error: Variable 'xxx' not found" pattern
    if re.search(r"Error: Variable '[^']+' not found", md_content):
        return True
    return False


def run_single_task(task_name: str, script_path: str, summarization_dir: Path, log_dir: Path, run_id: int) -> tuple[str, bool, str]:
    """Run comparison for a single task. Returns (task_name, success, message)."""
    # Skip if no summarization .md file exists (readonly check, no lock needed)
    summarization_files = list(summarization_dir.glob(f"*{task_name}*.md"))
    if not summarization_files:
        return (task_name, True, "Skipped (no summarization)")
    
    lock_file = log_dir / f".{task_name}.lock"
    
    try:
        # Acquire lock with 0 timeout - fail immediately if another process holds it
        with FileLock(lock_file, timeout=0):
            # Check AFTER acquiring lock to prevent race condition
            existing_md = list(log_dir.glob(f"*{task_name}*.md"))
            if existing_md:
                # Check if the .md file contains error messages that require regeneration
                md_content = existing_md[0].read_text()
                if has_regeneration_error(md_content):
                    # Delete the .md and .jsonl files, then regenerate
                    for md_file in existing_md:
                        md_file.unlink()
                        print(f"🗑️  Deleted (error found): {md_file.name}")
                    existing_jsonl = list(log_dir.glob(f"*{task_name}*.jsonl"))
                    for jsonl_file in existing_jsonl:
                        jsonl_file.unlink()
                        print(f"🗑️  Deleted (error found): {jsonl_file.name}")
                    # Continue to run the comparison (don't return)
                else:
                    return (task_name, True, f"Skipped (md exists): {existing_md[0].name}")
            else:
                existing_jsonl = list(log_dir.glob(f"*{task_name}*.jsonl"))
                if existing_jsonl:
                    # Delete orphan .jsonl files (no .md) and regenerate
                    for jsonl_file in existing_jsonl:
                        jsonl_file.unlink()
                        print(f"🗑️  Deleted (orphan jsonl, no md): {jsonl_file.name}")
                    # Continue to run the comparison (don't return)
            
            subprocess.run(
                [
                    sys.executable,
                    script_path,
                    "--task_name", task_name,
                    "--summarization_dir", str(summarization_dir),
                    "--run_id", str(run_id),
                    "--log_dir", str(log_dir)
                ],
                check=True,
            )
            return (task_name, True, "Completed")
    except Timeout:
        return (task_name, True, "Skipped (another process is handling this task)")
    except subprocess.CalledProcessError as e:
        return (task_name, False, f"Error: exit code {e.returncode}")


def run_for_tasks(
    task_names: list[str] | None,
    summarization_dir: str,
    log_dir: str,
    script_path: str,
    run_id: int,
    max_workers: int = 4,
) -> None:
    """Run the comparison script for each task name in parallel.
    
    Args:
        task_names: List of task names to analyze. If None or empty, 
                    extracts task names from all .md files in summarization_dir.
        summarization_dir: Directory containing Round 1 summarization .md files.
        log_dir: Directory to save comparison outputs.
        script_path: Path to the comparison script.
        run_id: Run ID for the analysis.
        max_workers: Number of parallel workers (default: 4).
    """
    log_dir = Path(log_dir)
    summarization_path = Path(summarization_dir)
    
    if not task_names:
        # Extract task names from .md files in summarization_dir
        md_files = list(summarization_path.glob("*.md"))
        task_names = []
        for md_file in md_files:
            # Extract task name from filename (assumes format: *_<task_name>_*.md)
            parts = md_file.stem.split("_")
            # Find the task name portion (typically after model and benchmark info)
            # Example: gpt-5_summarization_513_billion-word-imputation_2026-02-14_...
            if len(parts) >= 4:
                # Task name is at index 3 (0: model, 1: type, 2: benchmark, 3: task)
                task_name = parts[3]
                if task_name not in task_names:
                    task_names.append(task_name)
        task_names.sort()
        print(f"Found {len(task_names)} tasks from .md files in {summarization_dir}")
    
    # Cap workers to task count to avoid idle workers
    max_workers = min(max_workers, len(task_names))
    print(f"Running comparison analysis for {len(task_names)} tasks with {max_workers} workers...")
    
    results = {"success": [], "failed": [], "skipped": []}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_single_task, task, script_path, summarization_path, log_dir, run_id): task 
            for task in task_names
        }
        
        for future in as_completed(futures):
            task_name, success, message = future.result()
            if "Skipped" in message:
                results["skipped"].append(task_name)
                print(f"⏭️  {task_name}: {message}")
            elif success:
                results["success"].append(task_name)
                print(f"✅ {task_name}: {message}")
            else:
                results["failed"].append(task_name)
                print(f"❌ {task_name}: {message}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Summary: {len(results['success'])} completed, {len(results['skipped'])} skipped, {len(results['failed'])} failed")
    if results["failed"]:
        print(f"Failed tasks: {results['failed']}")
    print(f"\nComparisons saved to: {log_dir}")


if __name__ == "__main__":
    script_path = "/home/winnieyangwn/rlm/experiments/percentile/gpt5/gpt5_mle_comparison_v4.py"
    summarization_dir = "/checkpoint/maui_sft/winnieyangwn/rlm_dumps/summarization/mle-30/v4-2"
    log_dir = "/checkpoint/maui_sft/winnieyangwn/rlm_dumps/comparison/mle-30/v4-2"
    run_id = 514
    
    
    mle_30 = [
    "aptos2019-blindness-detection",
    "h-and-m-personalized-fashion-recommendations",
    "mlsp-2013-birds",
    "plant-pathology-2021-fgvc8",
    "uw-madison-gi-tract-image-segmentation",
    "hms-harmful-brain-activity-classification",
    "multi-modal-gesture-recognition",
    "smartphone-decimeter-2022",
    "ventilator-pressure-prediction",
    "billion-word-imputation",
    "hotel-id-2021-fgvc8",
    "new-york-city-taxi-fare-prediction",
    "spooky-author-identification",
    "whale-categorization-playground",
    "bms-molecular-translation",
    "hubmap-kidney-segmentation",
    "nfl-player-contact-detection",
    "stanford-covid-vaccine",
    "cassava-leaf-disease-classification",
    "imet-2020-fgvc7",
    "nomad2018-predict-transparent-conductors",
    "tensorflow2-question-answering",
    "champs-scalar-coupling",
    "jigsaw-unintended-bias-in-toxicity-classification",
    "osic-pulmonary-fibrosis-progression",
    "tweet-sentiment-extraction",
    "freesound-audio-tagging-2019",
    "kuzushiji-recognition",
    "petfinder-pawpularity-score",
    "us-patent-phrase-to-phrase-matching"
    ]
    mle_45 = [
    "aerial-cactus-identification",
    "denoising-dirty-documents",
    "detecting-insults-in-social-commentary",
    "dog-breed-identification",
    "dogs-vs-cats-redux-kernels-edition",
    "histopathologic-cancer-detection",
    "jigsaw-toxic-comment-classification-challenge",
    "leaf-classification",
    "plant-pathology-2020-fgvc7",
    "random-acts-of-pizza",
    "ranzcr-clip-catheter-line-classification",
    "siim-isic-melanoma-classification",
    "tabular-playground-series-dec-2021",
    "tabular-playground-series-may-2022",
    "text-normalization-challenge-english-language",
    "text-normalization-challenge-russian-language",
    "the-icml-2013-whale-challenge-right-whale-redux",
    "alaska2-image-steganalysis",
    "cdiscount-image-classification-challenge",
    "chaii-hindi-and-tamil-question-answering",
    "facebook-recruiting-iii-keyword-extraction",
    "google-quest-challenge",
    "herbarium-2020-fgvc7",
    "herbarium-2021-fgvc8",
    "herbarium-2022-fgvc9",
    "icecube-neutrinos-in-deep-ice",
    "inaturalist-2019-fgvc6",
    "iwildcam-2020-fgvc7",
    "learning-agency-lab-automated-essay-scoring-2",
    "lmsys-chatbot-arena",
    "seti-breakthrough-listen",
    "statoil-iceberg-classifier-challenge",
    "tensorflow-speech-recognition-challenge",
    "tgs-salt-identification-challenge",
    "3d-object-detection-for-autonomous-vehicles",
    "google-research-identify-contrails-reduce-global-warming",
    "iwildcam-2019-fgvc6",
    "predict-volcanic-eruptions-ingv-oe",
    "rsna-2022-cervical-spine-fracture-detection",
    "rsna-breast-cancer-detection",
    "rsna-miccai-brain-tumor-radiogenomic-classification",
    "siim-covid19-detection",
    "vesuvius-challenge-ink-detection",
    "vinbigdata-chest-xray-abnormalities-detection",
    "AI4Code"
    ]
    # mle_30_2=[ "us-patent-phrase-to-phrase-matching"
    #            "petfinder-pawpularity-score",
    #             ]
    # mle_30_8 = [
    #             # "cassava-leaf-disease-classification",
    #             # "whale-categorization-playground",
    #             "h-and-m-personalized-fashion-recommendations",
    #             # "kuzushiji-recognition",
    #             "spooky-author-identification",
    #             "billion-word-imputation",
    #             "tweet-sentiment-extraction",
    #             "hubmap-kidney-segmentation"
    # ]

    run_for_tasks(mle_30, summarization_dir, log_dir, script_path, run_id,  max_workers=20)
    # run_for_tasks(mle_30_2, summarization_dir, log_dir, script_path, run_id, max_workers=5)
