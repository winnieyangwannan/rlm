"""
Cross-tier contrastive comparison pipeline for MLE Bench rollouts using RLM.

This script analyzes patterns across all performance tiers in a single pass by:
1. Loading per-rollout summarization .md files from a specified directory
2. Building a unified DataFrame with all rollouts and their tier assignments
3. Using RLM to generate a contrastive analysis identifying what distinguishes
   each performance tier and the key transitions between them

Tiers (highest to lowest): 🥇 Medal → 🔵 Very High → 🟢 High → 🟡 Medium → 🔴 Low

Unlike v7 (which does pairwise tier comparisons), this version analyzes all tiers
together to build a holistic theory of what drives performance on the task.

Outputs:
  - Single .md report with cross-tier contrastive patterns
  - RLM .jsonl log for the analysis session

Requires: Completed summarization files (from gpt5_mle_summarization.py or similar)

Usage:
    python gpt5_mle_comparison_v6.py --task_name <task> --summarization_dir <dir>
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv

from rlm import RLM
from rlm.logger import RLMLogger

from experiments.percentile.gpt5.lesson_utils import (
    build_summaries_df,
    load_summarization_folder,
)

if TYPE_CHECKING:
    from rlm.core.types import RLMChatCompletion

load_dotenv()



# =============================================================================
# Configuration (defaults - can be overridden via CLI)
# =============================================================================
DEFAULT_CONFIG = {
    "run_id": 514,
    "model_name": "gpt-5",
    "job_name": "comparison",
    "task_name": "tweet-sentiment-extraction",  # Used in log file naming; does not affect which summarization files are loaded (those are determined by --summarization_dir and the presence of .md files)
    "log_dir": "/checkpoint/maui_sft/winnieyangwn/rlm_dumps/comparison/mle-30/v6",
    "summarization_dir": "/checkpoint/maui_sft/winnieyangwn/rlm_dumps/summarization/mle-30/v6",
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare MLE Bench solutions given summarization files")
    parser.add_argument("--summarization_dir", type=str, default=DEFAULT_CONFIG["summarization_dir"], help="Directory containing summarization .md files")
    parser.add_argument("--run_id", type=int, default=DEFAULT_CONFIG["run_id"], help="Run ID to analyze")
    parser.add_argument("--model", type=str, default=DEFAULT_CONFIG["model_name"], help="Model name to use")
    parser.add_argument("--job_name", type=str, default=DEFAULT_CONFIG["job_name"], help="Job name for logging")
    parser.add_argument("--task_name", type=str, default="tweet-sentiment-extraction", help="Task name to analyze")
    parser.add_argument("--log_dir", type=str, default=DEFAULT_CONFIG["log_dir"], help="Directory to save output logs")
    parser.add_argument("--max_depth", type=int, default=2, help="Max recursion depth for RLM")
    parser.add_argument("--max_iterations", type=int, default=20, help="Max iterations for RLM")
    parser.add_argument("--verbose", action="store_true", default=True, help="Enable verbose output")
    return parser.parse_args()


SCHEMA = """
================================================================================
AVAILABLE VARIABLES
================================================================================
  - `SUMMARIES`: DataFrame containing parsed rollout summaries (NOTE: use this instead of `context`, which is empty). Columns:
    - `rollout_id`: int - Rollout identifier
    - `score_percentile`: float - Score percentile (0-1)
    - `medal_earned`: str or None - Medal type (gold/silver/bronze) or None
    - `task_name`: str - Task name for the summarization
    - `tier`: str - Performance tier label (🥇 Medal, 🔵 Very High, 🟢 High, 🟡 Medium, 🔴 Low)
    - `task_analysis`: str - Flattened string with goal, task_type, data_modality, evaluation_metric, core_challenges, difficulty_factors
    - `summary`: str - 2-3 sentence overview of the solution
    - `full_summary`: str - **PRE-COMPUTED** full markdown summary combining all dimensions including: data_preprocessing, feature_engineering, model_selection, training_methodology, evaluation_and_submission, notable_implementation_details. Use this for passing to llm_query.
  - `pd`: pandas is already imported

================================================================================
AVAILABLE FUNCTIONS
================================================================================
  - `llm_query(prompt)`: Query a sub-LLM with a single prompt string. Returns response string.
  - `llm_query_batched(prompts)`: Query sub-LLM with multiple prompts in parallel. Pass list of strings, returns list of responses. **Preferred for efficiency.**
  - `FINAL_VAR(variable_name)`: Return a REPL variable as your final answer. Call only when analysis is complete.

================================================================================
PYTHON REPL CONSTRAINTS
================================================================================
  - **F-string limitation**: You CANNOT use backslash characters (`\\n`, `\\t`, etc.) directly inside f-string `{}` expressions. This causes a SyntaxError.
    - ❌ WRONG: `f"{separator.join(items)}"` where separator contains `\\n`
    - ❌ WRONG: `f"{'\\n'.join(items)}"`
    - ✅ CORRECT: Assign to a variable first: `newline = "\\n"; result = newline.join(items); f"{result}"`
    - ✅ CORRECT: Use `.format()` or `%` formatting: `"{}".format("\\n".join(items))`
    - ✅ CORRECT: Use concatenation: `"Header:\\n" + "\\n".join(items)`
"""



CONTRASTIVE_REFLECTION_TEMPLATE = """
## Goal
Analyze summaries for previous solutions and task analyses to identify patterns that distinguish performance tiers.

---

## Core Principle

Your job is NOT only to summarize what solutions did, but also build a DEEP THEORY of what makes solutions succeed or fail on this specific task, supported by evidence from the solutions. Try to connect every claim to a concrete property of the dataset, metric, or task structure. 
Remember that you are building a model of performance on this task — a mental model that could help PREDICT whether a new unseen solution would score high or low based on its design choices, given what you know about the data and metric.
Generic ML insights (e.g., "tune hyperparameters," "use cross-validation") without task-specific justification is not useful.


---

## REPL-First Workflow

**You are in a REPL environment.** You MUST execute code before producing any answer.


❌ **DO NOT** call `FINAL_VAR()` without first running REPL code  
❌ **DO NOT** produce a final answer in your first response  
✅ **DO** start by exploring the data with `print()` statements

---

## Performance Tiers

Use these tiers consistently throughout your analysis:

| Tier | Label | Criteria | Shorthand |
|------|-------|----------|-----------|
| 🥇 | **Medal** | Medal-winning solutions | `medal` |
| 🔵 | **Very High** | percentile ≥ 0.9, no medal | `very_high` |
| 🟢 | **High** | 0.7 ≤ percentile < 0.9 | `high` |
| 🟡 | **Medium** | 0.4 ≤ percentile < 0.7 | `medium` |
| 🔴 | **Low** | percentile < 0.4 | `low` |


---


## Analysis Structure

### 1. Tier Distribution Summary

Before diving into patterns, report the number of solutions in each tier. This contextualizes the analysis. 

### 2. Contrastive Pattern Analysis (Across Tiers)

Analyze patterns across the following dimensions:
  1. Data Preprocessing
  2. Feature Engineering
  3. Model Selection
  4. Training Methodology
  5. Evaluation & Submission
  6. Notable Implementation Details

**[Dimension Name]**

- **🥇 Medal**: [What they did, with count]
- **🔵 Very High (≥0.9)**: [What they did, with count]
- **🟢 High (0.7–0.9)**: [What they did, with count]
- **🟡 Medium (0.4–0.7)**: [What they did, with count]
- **🔴 Low (<0.4)**: [What they did, with count]
- **Key transitions**:
  - *Low → Medium*: [What changes move solutions out of the low tier?]
  - *Medium → High*: [What techniques unlock above-average performance?]
  - *High → Very High*: [What refinements push into the top tier?]
  - *Very High → 🥇 Medal*: [What separates near-medal from medal?]
- **Why do these patterns matter?** (Underlying reasoning tied to task/data/metric specifics)
- **Actionable recommendations by ambition level**:
  - *Target Medal*: [Advice]
  - *Target High+*: [Advice]
  - *Minimum viable*: [Advice to avoid Low tier]

---

## Returning Your Final Answer

After you have analyzed the data using REPL code:
1. Store your complete answer in a variable named `final_answer`
2. Call: `FINAL_VAR(final_answer)"""


def main() -> None:
    args = parse_args()

    # Validate required environment variables
    required_env_vars = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

    # =========================================================================
    # LOAD ROLLOUT SUMMARIZATION FOLDER
    # =========================================================================
    print("\n" + "=" * 80)
    print(f"Loading summarizations for task: {args.task_name}")
    print("=" * 80)
    
    summarization_folder = load_summarization_folder(args.summarization_dir, args.task_name)

    if summarization_folder is None:
        print("ERROR: Could not find summarization folder. Exiting.")
        return

    # =========================================================================
    # LOAD SUMMARIES DATAFRAME (directly from .md files)
    # =========================================================================
    summaries_df = build_summaries_df(summarization_folder, task_name=args.task_name)
    
    if summaries_df.empty:
        print("ERROR: No summaries found. Exiting.")
        return
    
    # Save to temp .jsonl for SETUP_CODE to load
    summaries_jsonl_path = summarization_folder / f"{args.model}_{args.job_name}_{args.run_id}_{args.task_name}_all_rollouts.jsonl"
    summaries_df.to_json(summaries_jsonl_path, orient="records", lines=True)
    print(f"Saved SUMMARIES to: {summaries_jsonl_path} ({len(summaries_df)} rows)")

    # =========================================================================
    # RLM INITIALIZATION 
    # =========================================================================

    logger = RLMLogger(
        log_dir=args.log_dir,
        file_name=f"{args.model}_{args.job_name}_{args.run_id}_{args.task_name}",
    )

    SETUP_CODE = f"""import pandas as pd

# Load summarization
SUMMARIES = pd.read_json('{summaries_jsonl_path}', lines=True)
print(f"Loaded {{len(SUMMARIES)}} summaries")
"""
    # =========================================================================
    # RLM CONTRASTIVE REFLECTION 
    # =========================================================================

    contrastive_reflection = RLM(
        backend="azure_openai",
        backend_kwargs={
            "model_name": args.model,
            "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
            "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
            "azure_deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            "api_version": "2025-03-01-preview",
        },
        environment="local",
        environment_kwargs={
            "setup_code": SETUP_CODE,
        },
        max_depth=args.max_depth,
        max_iterations=args.max_iterations,
        logger=logger,
        verbose=args.verbose,
    )


    root_prompt= f"{SCHEMA}\n\nQUESTION:\n{CONTRASTIVE_REFLECTION_TEMPLATE}"

    print(f"\nRunning contrastive reflection (max_depth={args.max_depth}, max_iterations={args.max_iterations})...\n")
    result = contrastive_reflection.completion(
        prompt="",
        root_prompt=root_prompt
    )
    # =========================================================================
    # SAVE
    # =========================================================================

    # Save result to .md file with same name and location as log file
    log_path = Path(logger.log_file_path)
    result_md_path = log_path.with_suffix(".md")
    with open(result_md_path, "w") as f:
        f.write(result.response)
    print(f"\nResult saved to: {result_md_path}")


if __name__ == "__main__":
    main()