"""
Contrastive comparison pipeline for MLE Bench rollouts using RLM.

This script analyzes rollout metadata directly:
1. Loads rollout metadata DataFrame (code_df)
2. Uses RLM to generate cross-tier contrastive analysis identifying patterns,
   critical failures, and unique high-performing approaches

Tiers (highest to lowest): 🥇 Super High → 🔵 Very High → 🟢 High → 🟡 Medium → 🔴 Low

Outputs:
  - Comparison report .md with tier patterns, failure analysis, and unique approaches
  - RLM .jsonl log for the analysis session

Usage:
    python gpt5_mle_comparison_v9.py --task_name <task>
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from rlm import RLM
from rlm.logger import RLMLogger

load_dotenv()


# =============================================================================
# Configuration (defaults - can be overridden via CLI)
# =============================================================================
DEFAULT_CONFIG = {
    "run_id": 514,
    "model_name": "gpt-5",
    "job_name": "comparison",
    "log_dir": "/checkpoint/maui_sft/winnieyangwn/rlm_dumps/comparison/mle-30/v9",
    "codebase_extensions": [".py", ".md", ".yaml"],
}



def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare MLE Bench solutions using rollout data")
    parser.add_argument("--run_id", type=int, default=DEFAULT_CONFIG["run_id"], help="Run ID to analyze")
    parser.add_argument("--model", type=str, default=DEFAULT_CONFIG["model_name"], help="Model name to use")
    parser.add_argument("--job_name", type=str, default=DEFAULT_CONFIG["job_name"], help="Job name for logging")
    parser.add_argument("--task_name", type=str, required=True, help="Task name to analyze (required)")
    parser.add_argument("--log_dir", type=str, default=DEFAULT_CONFIG["log_dir"], help="Directory to save output logs")
    parser.add_argument("--max_depth", type=int, default=2, help="Max recursion depth for RLM")
    parser.add_argument("--max_iterations", type=int, default=20, help="Max iterations for RLM")
    parser.add_argument("--verbose", action="store_true", default=True, help="Enable verbose output")
    return parser.parse_args()

CONTRASTIVE_REFLECTION_TEMPLATE = """
## Goal
Analyze past code solutions to extract **actionable, self-contained lessons** that a future agent can directly apply to match or exceed the best past performance on MLE Bench tasks.

## Core Principle
Your report is the **only context a future agent will have**. It must operate on two levels simultaneously:

- **Abstract (the "why")**: Build a mental model of *why* top solutions worked and *why* low-scoring solutions failed — through direct contrast. Identify the underlying reasoning, inductive biases, and problem-specific insights that separated winners from losers. A future agent should understand the logic well enough to adapt when conditions change.
- **Concrete (the "how")**: Provide exact decisions, numbers, and code so the future agent can replicate winning behavior without guesswork.

Neither level alone is sufficient. Principles without details are unactionable. Details without principles are brittle. Every major finding should have both.

## Contrastive Analysis Methodology
The primary analytical lens is **contrast**: for every claim about what works, ask what the failing solutions did instead and why that failed. The goal is not a list of good practices in isolation — it is a causal model of the performance gap.

When building your mental model, reason through:
- What assumption did low-scoring solutions make that was wrong?
- What did top solutions understand about the problem that low-scoring ones missed?
- Was the failure due to a wrong model of the data, the metric, the training dynamics, or the problem structure?
- Could a failing solution have been fixed with a single insight, or did it require a fundamentally different approach?

For each finding, structure your thinking as:
> **High scorers did X because [reason tied to problem structure]** ↔ **Low scorers did Y because [mistaken assumption or missed insight]** → **The gap is explained by [root cause]**

## Analysis Instructions
Compare high-performing solutions (top percentile / gold/silver medal) against low-performing ones (bottom percentile / no medal).

Your analysis must cover:

1. **Winning strategy overview** — Describe the end-to-end approach of the best solution(s) as a coherent narrative.
2. **Modeling choices** — What architectures, loss functions, or training paradigms appeared in top solutions but not bottom ones? Why were they well-suited to this problem's structure and evaluation metric?
3. **Feature engineering** — What preprocessing or feature transformations distinguished winners? What did losers overlook or get wrong?
4. **Hyperparameter patterns** — Exact values that worked, with reasoning for why they were appropriate.
5. **Failure modes & anti-patterns** — Root causes of low-scoring solutions, not just symptoms.
6. **Medal-winning differentiators** — The insight behind each technique that pushed solutions to medal level.

For each finding, support with **code snippets** from actual solutions. Prefer complete, runnable blocks over fragments.

## Output Format

### 0. TL;DR — Winning Recipe
A concise (5–10 bullet) checklist a future agent can follow to replicate top performance. Each bullet must be specific (e.g., "Use XGBoost with `n_estimators=1000, learning_rate=0.02, max_depth=6, subsample=0.8`" not "tune XGBoost carefully").

### 1. Contrastive Mental Model
 Build a mental model of the performance gap using direct contrast.

Structure it as a series of **contrastive pairs**, each addressing a key decision point in the solution space:

---
**[Decision Point, e.g., "Loss function choice"]**

| | High-scoring solutions | Low-scoring solutions |
|---|---|---|
| **What they did** | ... | ... |
| **Underlying assumption** | ... | ... |
| **Why it (worked / failed)** | ... | ... |

**Root cause of the gap**: [1–3 sentence causal explanation — what did top solutions understand that low ones didn't?]

---

After contrastive comparison, write a **Unified Mental Model** (1–2 paragraphs): synthesize the individual contrasts into a single coherent explanation of why top solutions won. What is the core insight that, if a future agent internalized it, would most improve their performance?

### 2. Problem Structure Analysis
Before diving into specifics, articulate:
- What kind of problem is this structurally? (e.g., tabular regression with heavy class imbalance, time-series with distribution shift, vision task with limited labels)
- What properties of the problem make certain approaches succeed or fail?
- What did winning solutions correctly identify about the problem that losing ones missed?

### 3. Winning Strategy Narrative
Walk through the best solution as a step-by-step playbook. Interleave the *reasoning* at each decision point with the *action* taken.

### 4. Modeling Choices
For each key modeling decision:
- **High scorers**: what they did and why it works (with code)
- **Low scorers**: what they did instead and why it failed (with code if illustrative)
- **Recommendation**: what the future agent should do

### 5. Feature Engineering
Same structure: high vs. low contrast → exact transformation code → recommendation.

### 6. Hyperparameters That Worked
Ready-to-use config block with inline comments explaining reasoning behind non-obvious values


### 7. Failure Modes to Avoid
For each anti-pattern:
- What was done and how often it appeared in low-scoring solutions
- Root cause (the *why*, not just the symptom) — what mistaken belief or oversight caused this?
- What top solutions did instead

### 8. Medal-Winning Differentiators ⭐
What separated medal solutions from merely good ones — the insight that motivated each technique, and what low-scoring solutions were missing that would have gotten them there.

---

## Returning Your Final Answer
After completing your analysis:
1. Store your complete report in a variable named `final_answer`
2. Call: `FINAL_VAR(final_answer)`

The report must be fully self-contained. A future agent reading it should understand not just *what* to do, but *why* the winning approach works and *why* the alternatives fail — well enough to perform at least as well as the best solution in this history.
"""

# Build prompt schema
round2_schema = f"""
================================================================================
AVAILABLE VARIABLES
================================================================================
  - `code_df`: DataFrame containing metadata of past solutions
    - `task_name`: Name of the MLE Bench task
    - `task_description`: Description of the task
    - `code`: The solution code
    - `percentile`: Performance score from 0 to 1 (1 = best, 0 = worst)
    - `medal`: Medal status (gold, silver, bronze, or None)
    - `valid_submission`: Whether the submission was valid
    - `eval_error_output`: Error output from evaluation (if any)
    - `eval_duration`: GPU execution duration for evaluation
    - `rollout_duration`: Total duration of the rollout
  - `pd`: pandas is already imported
"""

def validate_path(path: str, description: str) -> Path:
    """Validate that a path exists and return Path object."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    return p


def get_data_path(run_id: int) -> str:
    """Get the data path for a given run ID."""
    return f"/checkpoint/maui_sft/winnieyangwn/amaia_dumps/{run_id}/trajectories/{run_id}_metadata_code_only.jsonl"


def build_comparison_question() -> str:
    """Build the comparison question."""
    return CONTRASTIVE_REFLECTION_TEMPLATE


def main() -> None:
    args = parse_args()
    
    # Build paths
    data_path = get_data_path(args.run_id)
    validate_path(data_path, "Data file")

    # Validate required environment variables
    required_env_vars = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

    # =========================================================================
    # Compare solutions using rollout data
    # =========================================================================
    print("\n" + "=" * 80)
    print(f"Comparing solutions for task: {args.task_name}")
    print("=" * 80)


    task_suffix = f"_{args.task_name}" if args.task_name else ""
    logger = RLMLogger(
        log_dir=args.log_dir,
        file_name=f"{args.model}_{args.job_name}_{args.run_id}{task_suffix}"
    )

    SETUP_CODE = f"""
import pandas as pd

# Load rollout data
code_df = pd.read_json('{data_path}', lines=True)
{"code_df = code_df[code_df['task_name'] == '" + args.task_name + "']" if args.task_name else ""}
print(f"Loaded {{len(code_df)}} rollouts")
"""

    rlm = RLM(
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


    question = build_comparison_question()
    root_prompt = f"{round2_schema}\n\nQUESTION:\n{question}"

    print(f"\nRunning comparison (max_depth={args.max_depth}, max_iterations={args.max_iterations})...\n")
    result = rlm.completion(
        prompt="",
        root_prompt=root_prompt
    )

    # Save result to .md file with same name and location as log file
    log_path = Path(logger.log_file_path)
    result_md_path = log_path.with_name(log_path.stem + "_comparison_report.md")
    with open(result_md_path, "w") as f:
        f.write(result.response)
    print(f"\nResult saved to: {result_md_path}")


if __name__ == "__main__":
    main()