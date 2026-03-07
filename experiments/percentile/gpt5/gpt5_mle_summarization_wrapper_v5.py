"""
Wrapper script to run MLE Bench summarization analysis for multiple tasks.
"""

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from filelock import FileLock, Timeout


import re


def has_regeneration_error(md_file: Path) -> bool:
    """Check if the md file contains error messages that require regeneration.
    
    Synced with has_error_in_md() in gpt5_mle_summarization_v5.py
    """
    try:
        content = md_file.read_text().strip()
        # Check for empty file
        if not content:
            return True
        # Check for error prefix
        if content.startswith("Error:"):
            return True
        # Check for common REPL error patterns
        if "Traceback (most recent call last)" in content[:500]:
            return True
        # Check for timeout error
        if "Error: Request failed: timed out" in content:
            return True
        # Check for "Error: Variable 'xxx' not found" pattern
        if re.search(r"Error: Variable '[^']+' not found", content):
            return True
        return False
    except Exception:
        return True  # If we can't read the file, treat it as an error


def run_single_task(task_name: str, script_path: str, log_dir: Path) -> tuple[str, bool, str]:
    """Run summarization for a single task. Returns (task_name, success, message)."""
    lock_file = log_dir / f".{task_name}.lock"
    
    try:
        # Acquire lock with 0 timeout - fail immediately if another process holds it
        with FileLock(lock_file, timeout=0):
            # Check AFTER acquiring lock to prevent race condition
            # Use pattern matching new filename format: {task_name}_rollout{idx}_{timestamp}.md
            existing_md = list(log_dir.glob(f"{task_name}_rollout*_*.md"))
            if existing_md:
                # Check the most recent file for errors
                most_recent = max(existing_md, key=lambda p: p.stat().st_mtime)
                if has_regeneration_error(most_recent):
                    # Delete all md and jsonl files, then regenerate
                    for md_file in existing_md:
                        md_file.unlink()
                        print(f"🗑️  Deleted (error found): {md_file.name}")
                    existing_jsonl = list(log_dir.glob(f"*{task_name}*.jsonl"))
                    for jsonl_file in existing_jsonl:
                        jsonl_file.unlink()
                        print(f"🗑️  Deleted (error found): {jsonl_file.name}")
                    # Continue to regenerate below
                else:
                    return (task_name, True, f"Skipped (md exists): {most_recent.name}")
            else:
                existing_jsonl = list(log_dir.glob(f"*{task_name}*.jsonl"))
                if existing_jsonl:
                    # Delete orphan .jsonl files (no .md) and regenerate
                    for jsonl_file in existing_jsonl:
                        jsonl_file.unlink()
                        print(f"🗑️  Deleted (orphan jsonl, no md): {jsonl_file.name}")
                    # Continue to run the summarization (don't return)
            
            subprocess.run(
                [sys.executable, script_path,
                  "--account", "maui_sft",
                  "--run_id", "514",
                  "--log_dir", str(log_dir),
                  "--task_name", task_name],
                check=True,
            )
            return (task_name, True, "Completed")
    except Timeout:
        return (task_name, True, "Skipped (another process is handling this task)")
    except subprocess.CalledProcessError as e:
        return (task_name, False, f"Error: exit code {e.returncode}")


def run_for_tasks(task_names: list[str], max_workers: int = 4) -> None:
    """Run the summarization script for each task name in parallel."""
    # Cap workers to task count to avoid idle workers
    max_workers = min(max_workers, len(task_names))
    print(f"Running analysis for {len(task_names)} tasks with {max_workers} workers...")
    
    results = {"success": [], "failed": [], "skipped": []}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_single_task, task, script_path, log_dir): task 
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
    print(f"\nSummaries saved to: {log_dir}")


if __name__ == "__main__":


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
    mle_30_6 = ["freesound-audio-tagging-2019",
                "hotel-id-2021-fgvc8",
                "aptos2019-blindness-detection",
                "us-patent-phrase-to-phrase-matching"
                "whale-categorization-playground",
                "spooky-author-identification",

                ]
    mle_30_7 = ["kuzushiji-recognition",
                "hubmap-kidney-segmentation",
                "h-and-m-personalized-fashion-recommendations",
                "cassava-leaf-disease-classification",
                "billion-word-imputation",
                "multi-modal-gesture-recognition",
                "mlsp-2013-birds",]  
    script_path = "/home/winnieyangwn/rlm/experiments/percentile/gpt5/gpt5_mle_summarization_v5.py"
    log_dir = Path("/checkpoint/maui_sft/winnieyangwn/rlm_dumps/summarization/mle-30/v5")
     # 9 + 11 + 12 + 13
    run_for_tasks(mle_30_6, max_workers=11)  # Adjust max_workers based on API rate limits
