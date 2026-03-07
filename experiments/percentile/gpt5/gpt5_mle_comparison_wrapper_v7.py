"""
Wrapper script to run MLE Bench comparison analysis for multiple tasks (v7 format).

v7 format: summarization files are organized in task subfolders:
  {summarization_dir}/{task_name}/*.md

v7 changes:
- Uses df_pair for tier-wise comparison (pairs of adjacent tiers)
- Saves combined comparison report with all tier pairs assembled
"""

import re
import subprocess
import sys
import time
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


def run_single_task(
    task_name: str,
    script_path: str,
    summarization_dir: Path,
    log_dir: Path,
    run_id: int,
    model: str = "gpt-5",
    max_depth: int = 2,
    max_iterations: int = 20,
) -> tuple[str, bool, str, float]:
    """Run comparison for a single task. Returns (task_name, success, message, elapsed_seconds)."""
    # v7 format: summarization files are in task subfolders
    task_folder = summarization_dir / task_name
    if not task_folder.exists():
        return (task_name, True, "Skipped (no task folder)", 0.0)
    summarization_files = list(task_folder.glob("*.md"))
    if not summarization_files:
        return (task_name, True, "Skipped (no summarization)", 0.0)
    
    lock_file = log_dir / f".{task_name}.lock"
    
    try:
        # Acquire lock with 0 timeout - fail immediately if another process holds it
        with FileLock(lock_file, timeout=0):
            # Check AFTER acquiring lock to prevent race condition
            # v7: combined report has "_comparison_report.md" suffix
            existing_md = list(log_dir.glob(f"*{task_name}*_comparison_report.md"))
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
                    return (task_name, True, f"Skipped (md exists): {existing_md[0].name}", 0.0)
            else:
                existing_jsonl = list(log_dir.glob(f"*{task_name}*.jsonl"))
                if existing_jsonl:
                    # Delete orphan .jsonl files (no .md) and regenerate
                    for jsonl_file in existing_jsonl:
                        jsonl_file.unlink()
                        print(f"🗑️  Deleted (orphan jsonl, no md): {jsonl_file.name}")
                    # Continue to run the comparison (don't return)
            
            start_time = time.time()
            subprocess.run(
                [
                    sys.executable,
                    script_path,
                    "--task_name", task_name,
                    "--summarization_dir", str(summarization_dir),
                    "--run_id", str(run_id),
                    "--log_dir", str(log_dir),
                    "--model", model,
                    "--max_depth", str(max_depth),
                    "--max_iterations", str(max_iterations),
                ],
                check=True,
            )
            elapsed = time.time() - start_time
            return (task_name, True, "Completed", elapsed)
    except Timeout:
        return (task_name, True, "Skipped (another process is handling this task)", 0.0)
    except subprocess.CalledProcessError as e:
        return (task_name, False, f"Error: exit code {e.returncode}", 0.0)


def run_for_tasks(
    task_names: list[str] | None,
    summarization_dir: str,
    log_dir: str,
    script_path: str,
    run_id: int,
    max_workers: int = 4,
    model: str = "gpt-5",
    max_depth: int = 2,
    max_iterations: int = 20,
) -> None:
    """Run the comparison script for each task name in parallel.
    
    Args:
        task_names: List of task names to analyze. If None or empty, 
                    extracts task names from subfolders in summarization_dir.
        summarization_dir: Directory containing task subfolders with summarization .md files.
        log_dir: Directory to save comparison outputs.
        script_path: Path to the comparison script.
        run_id: Run ID for the analysis.
        max_workers: Number of parallel workers (default: 4).
        model: Model name to use for comparison (default: gpt-5).
        max_depth: Max recursion depth for RLM (default: 2).
        max_iterations: Max iterations for RLM (default: 20).
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    summarization_path = Path(summarization_dir)
    
    if not task_names:
        # v7 format: task names are subfolder names in summarization_dir
        task_folders = [f for f in summarization_path.iterdir() if f.is_dir()]
        task_names = sorted([f.name for f in task_folders if list(f.glob("*.md"))])
        print(f"Found {len(task_names)} tasks from subfolders in {summarization_dir}")
    
    # Cap workers to task count to avoid idle workers
    max_workers = min(max_workers, len(task_names))
    print(f"Running comparison analysis for {len(task_names)} tasks with {max_workers} workers...")
    
    results = {"success": [], "failed": [], "skipped": []}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                run_single_task, task, script_path, summarization_path, log_dir, run_id,
                model, max_depth, max_iterations
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
    print(f"\nComparisons saved to: {log_dir}")


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
    mle_test = ["freesound-audio-tagging-2019",]
    
    # v7 script path
    script_path = "/home/winnieyangwn/rlm/experiments/percentile/gpt5/gpt5_mle_comparison_v7.py"
    summarization_dir = "/checkpoint/maui_sft/winnieyangwn/rlm_dumps/summarization/mle-30/v6"
    log_dir = "/checkpoint/maui_sft/winnieyangwn/rlm_dumps/comparison/mle-30/v7"
    run_id = 514

    run_for_tasks(
        task_names=mle_30,
        summarization_dir=summarization_dir,
        log_dir=log_dir,
        script_path=script_path,
        run_id=run_id,
        max_workers=20,
        model="gpt-5",
        max_depth=2,
        max_iterations=20,
    )
