"""
Wrapper script to combine contrastive and success analyses into lesson.md files for multiple tasks (v6 format).

This script calls gpt5_mle_combine_lesson_v6.py for each task to combine:
  - Contrastive analysis from comparison_dir
  - Success analysis from success_dir
Into a single lesson.md file in output_dir.
"""

import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from filelock import FileLock, Timeout


def run_single_task(
    task_name: str,
    script_path: str,
    comparison_dir: Path,
    success_dir: Path,
    output_dir: Path,
    run_id: int,
    model: str = "gpt-5",
) -> tuple[str, bool, str, float]:
    """Run combine lesson for a single task. Returns (task_name, success, message, elapsed_seconds)."""
    # Check if at least one analysis exists for this task
    comparison_files = list(comparison_dir.glob(f"*{task_name}*.md"))
    success_files = list(success_dir.glob(f"*{task_name}*.md"))
    
    if not comparison_files and not success_files:
        return (task_name, True, "Skipped (no analysis files found)", 0.0)
    
    # Skip if lesson already exists
    lesson_file = output_dir / f"{task_name}_lesson.md"
    if lesson_file.exists():
        return (task_name, True, f"Skipped (lesson exists): {lesson_file.name}", 0.0)
    
    lock_file = output_dir / f".{task_name}.lock"
    
    try:
        # Acquire lock with 0 timeout - fail immediately if another process holds it
        with FileLock(lock_file, timeout=0):
            # Double-check after acquiring lock to prevent race condition
            if lesson_file.exists():
                return (task_name, True, f"Skipped (lesson exists): {lesson_file.name}", 0.0)
            
            start_time = time.time()
            subprocess.run(
                [
                    sys.executable,
                    script_path,
                    "--task_name", task_name,
                    "--run_id", str(run_id),
                    "--model", model,
                    "--comparison_dir", str(comparison_dir),
                    "--success_dir", str(success_dir),
                    "--output_dir", str(output_dir),
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
    task_names: list[str],
    comparison_dir: str,
    success_dir: str,
    output_dir: str,
    script_path: str,
    run_id: int,
    max_workers: int = 4,
    model: str = "gpt-5",
) -> None:
    """Run the combine lesson script for each task name in parallel.
    
    Args:
        task_names: List of task names to combine analyses for.
        comparison_dir: Directory containing contrastive analysis .md files.
        success_dir: Directory containing success analysis .md files.
        output_dir: Directory to save combined lesson.md files.
        script_path: Path to the combine lesson script.
        run_id: Run ID for the analysis.
        max_workers: Number of parallel workers (default: 4).
        model: Model name used in file naming (default: gpt-5).
    """
    comparison_path = Path(comparison_dir)
    success_path = Path(success_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Cap workers to task count to avoid idle workers
    max_workers = min(max_workers, len(task_names))
    print(f"Combining analyses for {len(task_names)} tasks with {max_workers} workers...")
    
    results = {"success": [], "failed": [], "skipped": []}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                run_single_task, task, script_path, comparison_path, success_path, 
                output_path, run_id, model
            ): task 
            for task in task_names
        }
        
        for future in as_completed(futures):
            task_name, success, message, elapsed = future.result()
            time_str = f" ({elapsed:.1f}s)" if elapsed > 0 else ""
            if "Skipped" in message:
                results["skipped"].append(task_name)
                print(f"⏭️  {task_name}: {message}")
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
    print(f"\nLessons saved to: {output_dir}")


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
    mle_test = ["tweet-sentiment-extraction",]
    script_path = "/home/winnieyangwn/rlm/experiments/percentile/gpt5/gpt5_mle_combine_lesson_v6.py"
    comparison_dir = "/checkpoint/maui_sft/winnieyangwn/rlm_dumps/comparison/mle-30/v6"
    success_dir = "/checkpoint/maui_sft/winnieyangwn/rlm_dumps/success_report/mle-30/v6"
    output_dir = "/checkpoint/maui_sft/winnieyangwn/rlm_dumps/lessons/mle-30/v6"
    run_id = 514

    run_for_tasks(
        task_names=mle_test,
        comparison_dir=comparison_dir,
        success_dir=success_dir,
        output_dir=output_dir,
        script_path=script_path,
        run_id=run_id,
        max_workers=20,
        model="gpt-5",
    )
