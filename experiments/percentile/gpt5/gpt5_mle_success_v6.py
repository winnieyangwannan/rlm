"""
Success analysis pipeline for MLE Bench high-performing solutions.

Analyzes top-tier rollouts to extract causal theories of what drives performance
on specific tasks. Uses RLM with Azure OpenAI to perform contrastive reflection
across successful solutions.

Tier hierarchy (in order of priority):
    - 🥇 Medal: Medal-winning solutions
    - 🔵 Very High: percentile ≥ 0.9, no medal
    - 🟢 High: 0.7 ≤ percentile < 0.9

Workflow:
    1. Load pre-computed summarization files from --summarization_dir
    2. Filter to highest available tier(s) — includes next tier if primary has only 1 rollout
    3. Run RLM contrastive reflection to produce a structured report:
       - Part 1: Per-solution analysis (key decisions, task insights)
       - Part 2: Cross-solution synthesis (shared success patterns)
       - Part 3: Valid divergences (where solutions legitimately differ)
       - Part 4: Actionable guidance (must-haves + high-variance design axes)
    4. Validate report completeness and retry if needed (up to 3 attempts)
    5. Save final report as markdown alongside the log file

Requires:
    - Completed summarization .md files (from gpt5_mle_summarization.py or similar)
    - Azure OpenAI credentials: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT

Example:
    python gpt5_mle_success_v6.py --task_name tweet-sentiment-extraction --model gpt-5
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
    "job_name": "success_report",
    "task_name": "tweet-sentiment-extraction",  # Used in log file naming; does not affect which summarization files are loaded (those are determined by --summarization_dir and the presence of .md files)
    "log_dir": "/checkpoint/maui_sft/winnieyangwn/rlm_dumps/success_analysis/mle-30/v6",
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


MEDAL_DEEP_DIVE_PROMPT = """
## Performance Tiers

| Tier | Label | Criteria | Shorthand |
|------|-------|----------|-----------|
| 🥇 | **Medal** | Medal-winning solutions | `medal` |
| 🔵 | **Very High** | percentile ≥ 0.9, no medal | `very_high` |
| 🟢 | **High** | 0.7 ≤ percentile < 0.9 | `high` |
| 🟡 | **Medium** | 0.4 ≤ percentile < 0.7 | `medium` |
| 🔴 | **Low** | percentile < 0.4 | `low` |

---

## Your Goal

Build a **causal theory** of what drives performance on this specific task — not a summary of what solutions did.

Every claim must be anchored to a concrete property of the **data, metric, or task structure**. The output should be specific enough that you could predict whether a new unseen solution would score high or low based on its design choices alone.

---

## Part 1 — Per-Solution Analysis

For each solution in the reference tier, analyze:

### 1.1 Key Decisions
Identify the 2–4 choices that most explain its high performance. For each:
- **What** was the choice? (e.g., specific model, feature type, preprocessing step, ensemble method, training strategy)
- **Why did it matter mechanistically?** Connect it to a specific property: the evaluation metric's behavior, a known data challenge (class imbalance, covariate shift, label noise, data modality constraints, etc.), or the task structure.
- **Counterfactual**: What would likely have happened without this choice? Be specific about the expected failure mode.

### 1.2 Task Insight
What did this solution understand about the task that lower-ranked solutions likely missed? 
- Reference specific core challenges or difficulty factors.
- Avoid vague claims like "better feature engineering" — name *which* features, *why* they captured signal the metric rewards, and *how* they interact with the data structure.

---

## Part 2 — Cross-Solution Synthesis

After analyzing individual solutions. Analyze what design choices appear relatively consistently across high-scoring solutions? These are the decisions that most determine performance on this task. Explain *why* each is load-bearing given the task's structure.


---

## Part 3 — Valid Divergences

Where do high-scoring solutions **differ** from each other? What does this tell you about which parts of the solution space have diverse valid approaches?


---

## Part 4 — Actionable Guidance for a New Agent

You are advising a future coding agent that will attempt this task from scratch. Its goal is not just to score high, but to produce solutions that are **meaningfully different** from each other while still being competitive.

Answer the following:

### 4.1 Must-Have Properties
What 2–3 design decisions are **non-negotiable** for achieving a high score on this task? For each, explain *why* omitting it reliably leads to failure, grounded in the task structure or metric.

### 4.2 High-Variance Design Axes
Identify 2–4 dimensions along which **valid high-scoring solutions can legitimately differ** — i.e., axes where multiple approaches work for different reasons. For each axis:
- Name the axis (e.g., model family, feature representation, training objective)
- Describe 2–3 distinct viable strategies along it
- Explain *why* this axis doesn't have a single dominant answer (e.g., tradeoff depends on data regime, metric tolerance, or compute budget)

These axes are where a diverse agent should explore rather than converge.

**Be concise. Avoid repetition and filler. Prioritize insight density over length.**

---

## OUTPUT REQUIREMENTS

**MANDATORY**: Your response MUST include ALL four parts in order:
1. Part 1 — Per-Solution Analysis (with sections 1.1 and 1.2 for each solution)
2. Part 2 — Cross-Solution Synthesis
3. Part 3 — Valid Divergences
4. Part 4 — Actionable Guidance (with sections 4.1 and 4.2)

Do NOT skip any parts. Do NOT jump directly to Part 4. Each part header must appear exactly as shown above.
"""


def validate_report_completeness(response: str) -> tuple[bool, list[str]]:
    """Validate that the report contains all required parts.
    
    Returns:
        tuple of (is_complete, missing_parts)
    """
    required_parts = [
        ("Part 1", "Part 1"),
        ("Part 2", "Part 2"),
        ("Part 3", "Part 3"),
        ("Part 4", "Part 4"),
    ]
    
    missing = []
    for pattern, name in required_parts:
        if pattern not in response:
            missing.append(name)
    
    return len(missing) == 0, missing


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
    # BUILD SUMMARIES DATAFRAME
    # =========================================================================
    summaries_df = build_summaries_df(summarization_folder, args.task_name)
    
    if summaries_df.empty:
        print("ERROR: No valid summaries found. Exiting.")
        return
    
    # Filter to the highest available tier (hierarchy: medal > very_high > high)
    # If highest tier has only 1 rollout, also include the next tier for more context
    tier_hierarchy = ["🥇 Medal", "🔵 Very High", "🟢 High"]
    selected_tiers = []
    
    # Find the highest tier that has rollouts
    primary_tier = None
    primary_tier_idx = None
    for idx, tier in enumerate(tier_hierarchy):
        if (summaries_df["tier"] == tier).any():
            primary_tier = tier
            primary_tier_idx = idx
            break
    
    if primary_tier is None:
        print(f"SKIP: No medal, very high, or high tier solutions found for task '{args.task_name}'. Skipping analysis.")
        print(f"  Tiers found: {summaries_df['tier'].value_counts().to_dict()}")
        return
    
    # Count rollouts in primary tier
    primary_tier_count = (summaries_df["tier"] == primary_tier).sum()
    selected_tiers.append(primary_tier)
    
    # If only 1 rollout in primary tier, include the next available tier
    if primary_tier_count == 1:
        for next_tier in tier_hierarchy[primary_tier_idx + 1:]:
            if (summaries_df["tier"] == next_tier).any():
                selected_tiers.append(next_tier)
                print(f"Primary tier '{primary_tier}' has only 1 rollout, also including '{next_tier}'")
                break
    
    # Filter DataFrame to include selected tiers
    summaries_df = summaries_df[summaries_df["tier"].isin(selected_tiers)].copy()
    tier_summary = ", ".join([f"{t}: {(summaries_df['tier'] == t).sum()}" for t in selected_tiers])
    print(f"Selected tiers: {selected_tiers} with {len(summaries_df)} total solutions ({tier_summary})")
    
    # Convert DataFrame to list of dicts for context_payload
    context_data = summaries_df.to_dict('records')
    print(f"Prepared context with {len(context_data)} rollouts")

    # Build context description for RLM prompt
    tier_str = ", ".join(selected_tiers)
    tier_dist = {t: (summaries_df['tier'] == t).sum() for t in selected_tiers}
    tier_dist_str = ", ".join([f"{t}: {c}" for t, c in tier_dist.items()])
    context_description = f"""================================================================================
AVAILABLE DATA
================================================================================
`context` is a Python list with {len(context_data)} rollout dicts for task '{args.task_name}'.

Tiers being analyzed: {tier_str}
Distribution: {tier_dist_str}

Each dict in `context` has these keys:
  - rollout_id: int - Rollout identifier
  - score_percentile: float - Score percentile (0-1)
  - medal_earned: str or None - Medal type (gold/silver/bronze) or None
  - task_name: str - Task name for the summarization
  - tier: str - Performance tier label (🥇 Medal, 🔵 Very High, 🟢 High, 🟡 Medium, 🔴 Low)
  - full_summary: str - Full markdown summary combining all dimensions

**IMPORTANT**: First create `SUMMARIES` DataFrame from `context`:
```python
SUMMARIES = pd.DataFrame(context)
```

Example access:
  - `SUMMARIES['tier'].unique()` → list of tier labels
  - `SUMMARIES.iloc[0]['full_summary']` → full summary text of first rollout

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

    # =========================================================================
    # RLM INITIALIZATION 
    # =========================================================================

    logger = RLMLogger(
        log_dir=args.log_dir,
        file_name=f"{args.model}_{args.job_name}_{args.run_id}_{args.task_name}",
    )

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
        environment_kwargs={},  # context_payload set by `prompt` arg in completion()
        max_depth=args.max_depth,
        max_iterations=args.max_iterations,
        logger=logger,
        verbose=args.verbose,
    )

    # Note: `prompt` becomes `context` in the REPL (via context_payload overwrite in rlm.py)
    # So we pass data as prompt, and instructions as root_prompt
    root_prompt = f"{context_description}\n\nQUESTION:\n{MEDAL_DEEP_DIVE_PROMPT}"

    print(f"\nRunning contrastive reflection (max_depth={args.max_depth}, max_iterations={args.max_iterations})...\n")
    
    # Retry logic to ensure complete report
    max_retries = 3
    result = None
    is_complete = False
    missing_parts = []
    
    for attempt in range(max_retries):
        result = contrastive_reflection.completion(
            prompt=context_data,  # This becomes `context` (list of dicts) in the REPL
            root_prompt=root_prompt
        )
        
        # Validate completeness
        is_complete, missing_parts = validate_report_completeness(result.response)
        
        if is_complete:
            print(f"Report validation passed on attempt {attempt + 1}")
            break
        else:
            print(f"Attempt {attempt + 1}/{max_retries}: Report incomplete. Missing: {missing_parts}")
            if attempt < max_retries - 1:
                print("Retrying...")
    
    if not is_complete:
        print(f"WARNING: Report still incomplete after {max_retries} attempts. Missing: {missing_parts}")
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