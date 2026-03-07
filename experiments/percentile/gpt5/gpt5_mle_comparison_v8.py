"""
Contrastive comparison pipeline for MLE Bench rollouts using RLM.

This script performs tier-based contrastive analysis on pre-computed solution
summaries to identify what distinguishes high-performing solutions from
low-performing ones on a given MLE Bench task.

Workflow:
    1. Load pre-computed summarization .md files from --summarization_dir
    2. Build a DataFrame of summaries with tier labels (🥇 Medal → ⚫ Lowest)
    3. Select the highest and lowest available tiers for comparison
    4. Use RLM to generate a structured contrastive report identifying:
       - Systematic differences across key dimensions (preprocessing, features,
         model selection, training, evaluation, implementation)
       - Actionable insights for improving lower-tier solutions

Inputs:
    - Summarization directory containing .md files (from gpt5_mle_summarization.py)
    - Task name to filter summaries

Outputs:
    - Comparison report .md with tier patterns and actionable recommendations
    - RLM .jsonl log for the analysis session

Usage:
    python gpt5_mle_comparison_v8.py --task_name <task> --summarization_dir <dir>

Example:
    python gpt5_mle_comparison_v8.py \\
        --task_name tweet-sentiment-extraction \\
        --summarization_dir /path/to/summarization/v6

Requires:
    - Completed summarization files (from gpt5_mle_summarization.py or similar)
    - Azure OpenAI credentials (AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT,
      AZURE_OPENAI_DEPLOYMENT)
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
    TIER_ORDER,
    TIER_REFERENCE,
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
    "log_dir": "/checkpoint/maui_sft/winnieyangwn/rlm_dumps/comparison/mle-30/v8",
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


TIER_PAIR_COMPARISON = """You are given `df_pair`, a dataframe containing all rollouts from the highest and lowest available tiers in a performance ranking. Each row represents one solution rollout, including its code and metadata.

Your task is to produce a structured contrastive report between these two extreme tiers.

---

## Core Principle

Do NOT merely summarize what each solution did. Instead, build a **deep theory** of what drives the performance gap between these two specific tiers, grounded in evidence from the solutions themselves.

Every claim must be tied to a concrete property of the task, dataset, or evaluation metric. Ask yourself: *why would this choice improve the score on this particular task?*

Generic ML advice (e.g., "tune hyperparameters," "use cross-validation") with no task-specific justification is not useful.

---

## Tier Reference

{TIER_REFERENCE}

The two tiers in `df_pair` are the highest and lowest available tiers from the hierarchy above. Refer to them by their actual tier labels throughout your report.

---

## Report Structure

### 1. Tier Distribution Summary

State which two tiers are being compared and how many rollouts each tier contains. Note any imbalance that may affect the reliability of observed patterns.

### 2. Contrastive Pattern Analysis

Identify what **systematically** differs between the two tiers across the dimensions below. Focus on patterns that appear across multiple solutions — not one-off quirks.

For each dimension, write:

**[Dimension Name]**
- **Lower tier pattern**: What do lower-tier solutions typically do here?
- **Higher tier pattern**: What do higher-tier solutions typically do here?
- **Key transition**: What is the most important change that separates them?
- **Why does it matter?** Explain the mechanism — tied specifically to the task structure, data properties, or evaluation metric.

Dimensions to cover:
1. Data Preprocessing
2. Feature Engineering
3. Model Selection
4. Training Methodology
5. Evaluation & Submission Strategy
6. Notable Implementation Details

### 3. Actionable Insights

List the 2–3 highest-leverage changes a lower-tier solution should make to reach the higher tier. For each, be specific: name the exact change, explain why it helps on this task, and flag any pitfalls to avoid.
"""



   

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
    
    # =========================================================================
    # BUILD HIGHEST VS LOWEST TIER COMPARISON
    # =========================================================================
    available_tiers = summaries_df["tier"].unique().tolist()
    # Sort by TIER_ORDER to find highest and lowest
    sorted_tiers = [t for t in TIER_ORDER if t in available_tiers]
    
    if len(sorted_tiers) < 2:
        print("ERROR: Not enough tiers to compare. Need at least 2 populated tiers.")
        return
    
    highest_tier = sorted_tiers[0]   # First in order (highest)
    lowest_tier = sorted_tiers[-1]   # Last in order (lowest)
    
    # Create comparison dataframe with highest and lowest tiers
    df_pair = summaries_df[summaries_df["tier"].isin([highest_tier, lowest_tier])].copy()
    
    print(f"\nComparing highest tier ({highest_tier}) vs lowest tier ({lowest_tier})")
    print(f"  Highest tier count: {len(df_pair[df_pair['tier'] == highest_tier])}")
    print(f"  Lowest tier count: {len(df_pair[df_pair['tier'] == lowest_tier])}")
    

    # =========================================================================
    # RLM CONTRASTIVE REFLECTION
    # =========================================================================
    
    tiers_in_pair = [highest_tier, lowest_tier]
    
    # Check if comparison_report.md already exists
    comparison_report_path = Path(args.log_dir) / f"{args.model}_{args.job_name}_{args.run_id}_{args.task_name}_comparison_report.md"
    if comparison_report_path.exists():
        print(f"\nSkipping: comparison report already exists: {comparison_report_path.name}")
        return
    
    print(f"\n{'=' * 80}")
    print(f"Processing comparison: {tiers_in_pair}")
    print(f"{'=' * 80}")
    
    # Convert df_pair to list of dicts for context_payload
    context_data = df_pair.to_dict('records')
    print(f"Prepared context with {len(context_data)} rollouts")
    
    # Build context description for RLM prompt
    tier_counts = df_pair["tier"].value_counts().to_dict()
    tier_counts_str = ", ".join([f"{tier}: {count}" for tier, count in tier_counts.items()])
    context_description = f"""================================================================================
AVAILABLE DATA
================================================================================
`context` is a Python list with {len(df_pair)} rollout dicts for task '{args.task_name}'.

Tiers being compared: {tiers_in_pair}
Distribution: {tier_counts_str}

Each dict in `context` has these keys:
  - rollout_id: int - Rollout identifier
  - score_percentile: float - Score percentile (0-1)
  - medal_earned: str or None - Medal type (gold/silver/bronze) or None
  - task_name: str - Task name for the summarization
  - tier: str - Performance tier label (🥇 Medal, 🔵 Very High, 🟢 High, 🟡 Medium, 🔴 Low, ⚫ Lowest)
  - full_summary: str - Full markdown summary combining all dimensions

**IMPORTANT**: First create `df_pair` from `context`:
```python
df_pair = pd.DataFrame(context)
```

Example access:
  - `df_pair['tier'].unique()` → list of tier labels
  - `df_pair[df_pair['tier'] == '🔴 Low']` → all Low tier rollouts
  - `df_pair.iloc[0]['full_summary']` → full summary text of first rollout

`pd` (pandas) is already imported.


================================================================================
PYTHON REPL CONSTRAINTS
================================================================================
  - **F-string limitation**: You CANNOT use backslash characters (`\\n`, `\\t`, etc.) directly inside f-string `{{}}` expressions. This causes a SyntaxError.
    - ❌ WRONG: `f"{{separator.join(items)}}"` where separator contains `\\n`
    - ❌ WRONG: `f"{{'\\n'.join(items)}}"`
    - ✅ CORRECT: Assign to a variable first: `newline = "\\n"; result = newline.join(items); f"{{result}}"`
    - ✅ CORRECT: Use `.format()` or `%` formatting: `"{{}}".format("\\n".join(items))`
    - ✅ CORRECT: Use concatenation: `"Header:\\n" + "\\n".join(items)`
"""
    
    logger = RLMLogger(
        log_dir=str(args.log_dir),
        file_name=f"{args.model}_{args.job_name}_{args.run_id}_{args.task_name}_comparison",
    )

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
        environment_kwargs={},  # context_payload set by `prompt` arg in completion()
        max_depth=args.max_depth,
        max_iterations=args.max_iterations,
        logger=logger,
        verbose=args.verbose,
    )

    # Note: `prompt` becomes `context` in the REPL (via context_payload overwrite in rlm.py)
    # So we pass data as prompt, and instructions as root_prompt
    tier_pair_prompt = TIER_PAIR_COMPARISON.format(TIER_REFERENCE=TIER_REFERENCE)
    root_prompt = f"{context_description}\n\nQUESTION:\n{tier_pair_prompt}"

    print(f"\nRunning contrastive reflection (max_depth={args.max_depth}, max_iterations={args.max_iterations})...\n")
    result = contrastive_reflection.completion(
        prompt=context_data,  # This becomes `context` (list of dicts) in the REPL
        root_prompt=root_prompt
    )

    # Save comparison report
    with open(comparison_report_path, "w") as f:
        f.write(result.response)
    print(f"\nComparison report saved to: {comparison_report_path}")


if __name__ == "__main__":
    main()