"""
Combine contrastive analysis and success analysis into a single lesson.md file.

This script:
1. Loads contrastive analysis from comparison output directory
2. Loads success analysis from success_analysis output directory  
3. Combines them into a single lesson.md with two sections
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path


# =============================================================================
# Configuration (defaults - can be overridden via CLI)
# =============================================================================
DEFAULT_CONFIG = {
    "run_id": 514,
    "model_name": "gpt-5",
    "task_name": "tweet-sentiment-extraction",
    "comparison_dir": "/checkpoint/maui_sft/winnieyangwn/rlm_dumps/comparison/mle-30/v6",
    "success_dir": "/checkpoint/maui_sft/winnieyangwn/rlm_dumps/success/mle-30/v6",
    "output_dir": "/checkpoint/maui_sft/winnieyangwn/rlm_dumps/lessons/mle-30/v6",
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Combine contrastive and success analyses into lesson.md")
    parser.add_argument("--task_name", type=str, default=DEFAULT_CONFIG["task_name"], help="Task name to combine analyses for")
    parser.add_argument("--run_id", type=int, default=DEFAULT_CONFIG["run_id"], help="Run ID")
    parser.add_argument("--model", type=str, default=DEFAULT_CONFIG["model_name"], help="Model name")
    parser.add_argument("--comparison_dir", type=str, default=DEFAULT_CONFIG["comparison_dir"], help="Directory containing contrastive analysis .md files")
    parser.add_argument("--success_dir", type=str, default=DEFAULT_CONFIG["success_dir"], help="Directory containing success analysis .md files")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_CONFIG["output_dir"], help="Directory to save combined lesson.md")
    return parser.parse_args()


def find_latest_analysis_file(
    directory: Path, 
    task_name: str,
    model: str = "gpt-5",
    job_name: str = "comparison",
    run_id: int = 514,
) -> Path | None:
    """
    Find the latest analysis .md file for a given task.
    
    File naming pattern: {model}_{job_name}_{run_id}_{task_name}_{timestamp}_{hash}.md
    Example: gpt-5_comparison_514_tweet-sentiment-extraction_2026-02-20_09-18-43_774bbde4.md
    """
    if not directory.exists():
        print(f"Warning: Directory does not exist: {directory}")
        return None
    
    # Pattern to match analysis files for this task
    pattern = f"{model}_{job_name}_{run_id}_{task_name}_*.md"
    matching_files = list(directory.glob(pattern))
    
    if not matching_files:
        print(f"Warning: No matching files found in {directory} for pattern: {pattern}")
        return None
    
    # Sort by modification time (most recent first)
    matching_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    
    return matching_files[0]


def load_contrastive_analysis(
    comparison_dir: str | Path,
    task_name: str,
    model: str = "gpt-5",
    run_id: int = 514,
) -> str | None:
    """
    Load the latest contrastive analysis for a task.
    
    Args:
        comparison_dir: Directory containing comparison analysis .md files
        task_name: Task name to load analysis for
        model: Model name used in file naming
        run_id: Run ID used in file naming
    
    Returns:
        Content of the contrastive analysis file, or None if not found
    """
    comparison_path = Path(comparison_dir)
    
    analysis_file = find_latest_analysis_file(
        directory=comparison_path,
        task_name=task_name,
        model=model,
        job_name="comparison",
        run_id=run_id,
    )
    
    if analysis_file is None:
        return None
    
    print(f"Loading contrastive analysis from: {analysis_file}")
    with open(analysis_file) as f:
        return f.read()


def load_success_analysis(
    success_dir: str | Path,
    task_name: str,
    model: str = "gpt-5",
    run_id: int = 514,
) -> str | None:
    """
    Load the latest success analysis for a task.
    
    Args:
        success_dir: Directory containing success analysis .md files
        task_name: Task name to load analysis for
        model: Model name used in file naming
        run_id: Run ID used in file naming
    
    Returns:
        Content of the success analysis file, or None if not found
    """
    success_path = Path(success_dir)
    
    analysis_file = find_latest_analysis_file(
        directory=success_path,
        task_name=task_name,
        model=model,
        job_name="success_report",
        run_id=run_id,
    )
    
    if analysis_file is None:
        return None
    
    print(f"Loading success analysis from: {analysis_file}")
    with open(analysis_file) as f:
        return f.read()


def combine_lessons(
    task_name: str,
    contrastive_analysis: str | None,
    success_analysis: str | None,
) -> str:
    """
    Combine contrastive and success analyses into a single lesson markdown.
    
    Args:
        task_name: Task name for the lesson
        contrastive_analysis: Content of contrastive analysis (or None)
        success_analysis: Content of success analysis (or None)
    
    Returns:
        Combined lesson markdown content
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    lesson = f"""# Lesson: {task_name}

Generated: {timestamp}

---

"""
    
    # Section 1: Contrastive Analysis
    lesson += "## 1. Contrastive Analysis\n\n"
    if contrastive_analysis:
        lesson += contrastive_analysis
    else:
        lesson += "*No contrastive analysis available for this task.*\n"
    
    lesson += "\n\n---\n\n"
    
    # Section 2: Success Analysis
    lesson += "## 2. Success Analysis\n\n"
    if success_analysis:
        lesson += success_analysis
    else:
        lesson += "*No success analysis available for this task.*\n"
    
    return lesson


def main() -> None:
    args = parse_args()
    
    print("\n" + "=" * 80)
    print(f"Combining analyses for task: {args.task_name}")
    print("=" * 80)
    
    # Load contrastive analysis
    contrastive_analysis = load_contrastive_analysis(
        comparison_dir=args.comparison_dir,
        task_name=args.task_name,
        model=args.model,
        run_id=args.run_id,
    )
    
    # Load success analysis
    success_analysis = load_success_analysis(
        success_dir=args.success_dir,
        task_name=args.task_name,
        model=args.model,
        run_id=args.run_id,
    )
    
    # Check if at least one analysis exists
    if contrastive_analysis is None and success_analysis is None:
        print("ERROR: No analyses found for this task. Exiting.")
        return
    
    # Combine into lesson
    lesson_content = combine_lessons(
        task_name=args.task_name,
        contrastive_analysis=contrastive_analysis,
        success_analysis=success_analysis,
    )
    
    # Save lesson
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    lesson_file = output_path / f"{args.task_name}_lesson.md"
    with open(lesson_file, "w") as f:
        f.write(lesson_content)
    
    print(f"\nLesson saved to: {lesson_file}")
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary:")
    print(f"  - Contrastive analysis: {'✅ Loaded' if contrastive_analysis else '❌ Not found'}")
    print(f"  - Success analysis: {'✅ Loaded' if success_analysis else '❌ Not found'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
