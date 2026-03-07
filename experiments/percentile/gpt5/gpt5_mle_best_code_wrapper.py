"""
Wrapper script to extract best code solutions for multiple tasks.

This script runs gpt5_mle_best_code.py for each task in parallel,
extracting the code from the rollout with the highest score.
"""

import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from filelock import FileLock, Timeout


def run_single_task(
    task_name: str,
    script_path: str,
    output_dir: Path,
    run_id: str,
    model: str = "gpt5",
    job_name: str = "code",
    group_name: str = "maui_sft",
) -> tuple[str, bool, str, float]:
    """Run best code extraction for a single task.
    
    Returns:
        Tuple of (task_name, success, message, elapsed_seconds)
    """
    lock_file = output_dir / f".{task_name}.lock"
    
    try:
        # Acquire lock with 0 timeout - fail immediately if another process holds it
        with FileLock(lock_file, timeout=0):
            # Check if output already exists
            existing_md = list(output_dir.glob(f"*{task_name}*_best.md"))
            if existing_md:
                return (task_name, True, f"Skipped (exists): {existing_md[0].name}", 0.0)
            
            start_time = time.time()
            env = os.environ.copy()
            env["PYTHONPATH"] = "/home/winnieyangwn/rlm:" + env.get("PYTHONPATH", "")
            
            subprocess.run(
                [
                    sys.executable,
                    script_path,
                    "--task_name", task_name,
                    "--output_dir", str(output_dir),
                    "--run_id", run_id,
                    "--model", model,
                    "--job_name", job_name,
                    "--group_name", group_name,
                ],
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


def run_for_tasks(
    task_names: list[str],
    output_dir: str,
    script_path: str,
    run_id: str,
    max_workers: int = 4,
    model: str = "gpt5",
    job_name: str = "code",
    group_name: str = "maui_sft",
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
                run_single_task, task, script_path, output_path, run_id,
                model, job_name, group_name
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
    
    mle_test = ["freesound-audio-tagging-2019"]
    mle_v4_5_leap= [
                "h-and-m-personalized-fashion-recommendations",
                "kuzushiji-recognition",
                "whale-categorization-playground",
                "hotel-id-2021-fgvc8",
                "hubmap-kidney-segmentation",
                ]

    mle_30_6 = ["mlsp-2013-birds",
                "multi-modal-gesture-recognition",
                "tweet-sentiment-extraction",
                "nomad2018-predict-transparent-conductors",
                "spooky-author-identification",
                "petfinder-pawpularity-score",
            ]
    mle_v4_4_improve= [
                   "cassava-leaf-disease-classification",
                   "billion-word-imputation",
                   "freesound-audio-tagging-2019",
                   "us-patent-phrase-to-phrase-matching"
                   ]
    mle_30_15 = [
        "aptos2019-blindness-detection",
        "plant-pathology-2021-fgvc8",
        "uw-madison-gi-tract-image-segmentation",
        "hms-harmful-brain-activity-classification",
        "smartphone-decimeter-2022",
        "ventilator-pressure-prediction",
        "new-york-city-taxi-fare-prediction",
        "bms-molecular-translation",
        "nfl-player-contact-detection",
        "stanford-covid-vaccine",
        "imet-2020-fgvc7",
        "tensorflow2-question-answering",
        "champs-scalar-coupling",
        "jigsaw-unintended-bias-in-toxicity-classification",
        "osic-pulmonary-fibrosis-progression",
    ]

    mle_30_r2_7 = [        
                    "hubmap-kidney-segmentation",
                    "h-and-m-personalized-fashion-recommendations",
                    "spooky-author-identification",
                    "kuzushiji-recognition",
                    "mlsp-2013-birds",
                    "whale-categorization-playground",
                    "hotel-id-2021-fgvc8",
                 ]
    
    # Configuration
    script_path = "/home/winnieyangwn/rlm/experiments/percentile/gpt5/gpt5_mle_best_code.py"

    # Round 1
    # task_names = mle_30
    # output_dir = "/checkpoint/maui_sft/winnieyangwn/rlm_dumps/best_code/mle_30_r1"
    # run_id = "514"
    # group_name="maui_sft"

    # Round 2
    task_names = mle_30_r2_7
    output_dir = "/checkpoint/agentic-models/winnieyangwn/rlm_dumps/best_code/mle_30_r2"
    run_id = "524_code"
    group_name="agentic-models"

    # Round 3
    # task_names = mle_30_r2_7 
    # output_dir = "/checkpoint/agentic-models/winnieyangwn/rlm_dumps/best_code/mle_30_r3"
    # run_id = "524_code_mle_30_r2_7"
    # group_name="agentic-models"


    run_for_tasks(
        task_names=task_names,
        output_dir=output_dir,
        script_path=script_path,
        run_id=run_id,
        max_workers=8,
        model="gpt5",
        job_name="code",
        group_name=group_name,
    )
