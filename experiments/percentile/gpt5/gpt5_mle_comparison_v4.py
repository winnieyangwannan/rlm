"""
Two-stage contrastive comparison pipeline for MLE Bench rollouts using RLM.

This script performs "Round 2" analysis on a pre-generated summarization file:
1. Loads a single Round 1 summarization .md file (markdown string from prior run)
2. Loads original rollout metadata for additional context (rollout_df)
3. Uses RLM to generate cross-tier contrastive analysis identifying patterns,
   critical failures, and unique high-performing approaches

Tiers (highest to lowest): 🥇 Super High → 🔵 Very High → 🟢 High → 🟡 Medium → 🔴 Low

Key difference from v6:
  - v4: Takes a single pre-generated summarization .md file as input (two-stage pipeline)
  - v6: Loads individual per-rollout .md files and builds DataFrame directly (single-stage)

Outputs:
  - Comparison report .md with tier patterns, failure analysis, and unique approaches
  - RLM .jsonl log for the analysis session

Requires: A completed Round 1 summarization .md file (from gpt5_mle_summarization.py)

Usage:
    python gpt5_mle_comparison_v4.py --task_name <task> --summarization_dir <dir>
"""

import argparse
import json
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
    "run_id": 513,
    "model_name": "gpt-5",
    "job_name": "comparison",
    "log_dir": "/checkpoint/maui_sft/winnieyangwn/rlm_dumps/comparison/mle-30/v4-2",
    "codebase_extensions": [".py", ".md", ".yaml"],
}



def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare MLE Bench solutions given Round 1 summarization")
    parser.add_argument("--summarization_dir", type=str, default="/checkpoint/maui_sft/winnieyangwn/rlm_dumps/summarization/mle-30", help="Directory containing Round 1 summarization .md files")
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
Analyze summaries for previous solutions and task analyses to identify:
1. **Patterns that distinguish performance tiers** (optimization)
2. **Critical failures that cause suboptimal results** (failure prevention)
3. **Unique top-performing approaches** (diversity preservation)

---

## Core Principle

Your job is NOT only to summarize what solutions did, but also build a DEEP THEORY of what makes solutions succeed or fail on this specific task, supported by evidence from the solutions. Try to connect every claim to a concrete property of the dataset, metric, or task structure. 
Remember that you are building a model of performance on this task — a mental model that could help PREDICT whether a new unseen solution would score high or low based on its design choices, given what you know about the data and metric.
Generic ML insights (e.g., "tune hyperparameters," "use cross-validation") without task-specific justification is not useful.


---

## Available Variables:
- `summarization`: Markdown string containing per-solution summaries across all rollouts. 
- `rollout_df`: Original rollout data (for additional context if needed)

---

## REPL-First Workflow

**You are in a REPL environment.** You MUST execute code before producing any answer.

### Required Workflow:
1. **FIRST**: Run exploratory code (e.g., `print(summarization[:3000])`)
2. **THEN**: Analyze incrementally using print statements and `llm_query()` if needed
3. **FINALLY**: Store complete answer in `final_answer` variable and call `FINAL_VAR(final_answer)`

### Available Functions:
- `llm_query(prompt)` - Call the LLM for sub-analysis
- `llm_query_batched(prompts)` - Batch multiple LLM calls
- `FINAL_VAR(variable_name)` - Return your final answer (call ONLY when done)

❌ **DO NOT** call `FINAL_VAR()` without first running REPL code  
❌ **DO NOT** produce a final answer in your first response  
✅ **DO** start by exploring the data with `print()` statements

---

## Performance Tiers

Use these tiers consistently throughout your analysis:

| Tier | Label | Criteria | Shorthand |
|------|-------|----------|-----------|
| 🥇 | **Super High** | Medal-winning solutions | `medal` |
| 🔵 | **Very High** | percentile ≥ 0.9, no medal | `very_high` |
| 🟢 | **High** | 0.7 < percentile < 0.9 | `high` |
| 🟡 | **Medium** | 0.4 ≤ percentile ≤ 0.7 | `medium` |
| 🔴 | **Low** | percentile < 0.4 | `low` |


---


## Analysis Structure

### 1. Tier Distribution Summary

Before diving into patterns, report the number of solutions in each tier. This contextualizes the analysis. 
### 2. Contrastive Pattern Analysis (Across Tiers)

For each key dimension (Data Preprocessing, Feature Engineering, Model Selection, Training Methodology, Evaluation & Submission, Notable Implementation Details):

**[Dimension Name]**

- **🥇 Super High (medal)**: [What they did, with frequency/count]
- **🔵 Very High (≥0.9)**: [What they did, with frequency/count]
- **🟢 High (0.7–0.9)**: [What they did, with frequency/count]
- **🟡 Medium (0.4–0.7)**: [What they did, with frequency/count]
- **🔴 Low (<0.4)**: [What they did, with frequency/count]
- **Key transitions**:
  - *Low → Medium*: [What changes move solutions out of the low tier?]
  - *Medium → High*: [What techniques unlock above-average performance?]
  - *High → Very High*: [What refinements push into the top tier?]
  - *Very High → Medal*: [What separates near-medal from medal?]
- **Why do these patterns matter?** (Underlying reasoning tied to task/data/metric specifics)
- **Actionable recommendations by ambition level**:
  - *Target Medal*: [Advice]
  - *Target High+*: [Advice]
  - *Minimum viable*: [Advice to avoid Low tier]

### 3. Critical Failure Analysis

Identify patterns that are **strongly associated with 🔴 Low tier** solutions:
- What mistakes or omissions produce low scores?
- Are there "trap" approaches that seem reasonable but fail on this specific task?
- What are the most important things to get right to avoid catastrophic failure?

### 4. Unique High-Performing Approaches (Diversity Preservation)

**GOAL: Identify rare but effective techniques that could be lost in convergence to common patterns.**

Examine **🥇 Super High and 🔵 Very High solutions** for techniques that are:
- ✅ **Rare**: used by few solutions (not common even among high-scoring solutions)
- ✅ **Effective**: plausibly contributed to the solution's top-tier performance

For each unique approach, note:
- Which tier(s) used it and how many solutions
- Why it likely helps (mechanistic reasoning, not just correlation)
- Risk/complexity of adopting it

---

## Returning Your Final Answer

After you have analyzed the data using REPL code:
1. Store your complete answer in a variable named `final_answer`
2. Call: `FINAL_VAR(final_answer)`"""

# Build Round 2 prompt
round2_schema = f"""
================================================================================
AVAILABLE VARIABLES
================================================================================
  - `summarization`: String containing the full summarization of past solutions
    - Contains task analysis and individual solution summaries
    - Each solution summary includes: the solution's score percentile, medal status, and details across six dimensions: data preprocessing, feature engineering, model selection, training methodology, evaluation & submission, and notable implementation details.
  - `rollout_df`: Original rollout DataFrame (for additional context if needed)
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
    return f"/checkpoint/maui_sft/winnieyangwn/amaia_dumps/{run_id}/trajectories/{run_id}_metadata.jsonl"


def load_summarization_md(summarization_dir: str, task_name: str) -> Path | None:
    """Find Round 1 summarization .md file path for a given task.
    
    Args:
        summarization_dir: Directory containing summarization .md files
        task_name: Task name to find the .md file for
        
    Returns:
        Path to the .md file, or None if not found
    """
    summarization_path = Path(summarization_dir)
    if not summarization_path.exists():
        print(f"Summarization directory not found: {summarization_dir}")
        return None
    
    # Find .md files matching the task_name pattern
    pattern = f"*_{task_name}_*.md"
    md_files = list(summarization_path.glob(pattern))
    
    if not md_files:
        print(f"No .md file found for task: {task_name}")
        print(f"Searched in: {summarization_dir}")
        print(f"Pattern: {pattern}")
        return None
    
    if len(md_files) > 1:
        print(f"Warning: Found {len(md_files)} .md files for task {task_name}, using most recent")
        md_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    md_file = md_files[0]
    print(f"Found summarization file: {md_file}")
    return md_file


def prepare_round2_context(round1_analysis: str, output_file: str) -> str:
    """Save Round 1 analysis to file for Round 2 to load.
    
    Args:
        round1_analysis: The markdown/text analysis from Round 1
        output_file: Path to save the analysis
        
    Returns:
        Path to the saved file
    """
    with open(output_file, "w") as f:
        f.write(round1_analysis)
    print(f"Saved Round 1 analysis to: {output_file}")
    return output_file


def build_round2_question() -> str:
    """Build the Round 2 comparison question."""
    return  CONTRASTIVE_REFLECTION_TEMPLATE


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
    # LOAD ROLLOUT SUMMARIZATION .md FILE
    # =========================================================================
    print("\n" + "=" * 80)
    print(f"Loading Round 1 summarization for task: {args.task_name}")
    print("=" * 80)
    
    summarization_path = load_summarization_md(args.summarization_dir, args.task_name)

    if summarization_path is None:
        print("ERROR: Could not find Round 1 summarization file. Exiting.")
        return

    # =========================================================================
    # ROUND 2: Compare solutions using summarization and rollout data
    # =========================================================================
    print("\n" + "=" * 80)
    print("ROUND 2: Comparing solutions...")
    print("=" * 80)


    task_suffix = f"_{args.task_name}" if args.task_name else ""
    logger_round2 = RLMLogger(
        log_dir=args.log_dir,
        file_name=f"{args.model}_{args.job_name}_{args.run_id}{task_suffix}"
    )

    SETUP_CODE = f"""
import pandas as pd

# Load original rollout data
rollout_df = pd.read_json('{data_path}', lines=True)
{"rollout_df = rollout_df[rollout_df['task_name'] == '" + args.task_name + "']" if args.task_name else ""}
print(f"Loaded {{len(rollout_df)}} rollouts")

# Load rollout summarization (markdown text)
with open('{summarization_path}') as f:
    summarization = f.read()
print(f"Loaded {{len(summarization)}} chars of summarization")
"""

    rlm_round2 = RLM(
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
        logger=logger_round2,
        verbose=args.verbose,
    )


    question_round2 = build_round2_question()
    root_prompt_round2 = f"{round2_schema}\n\nQUESTION:\n{question_round2}"

    print(f"\nRunning Round 2 (max_depth={args.max_depth}, max_iterations={args.max_iterations})...\n")
    result_round2 = rlm_round2.completion(
        prompt="",
        root_prompt=root_prompt_round2
    )

    # Save result to .md file with same name and location as log file
    log_path = Path(logger_round2.log_file_path)
    result_md_path = log_path.with_name(log_path.stem + "_comparison_report.md")
    with open(result_md_path, "w") as f:
        f.write(result_round2.response)
    print(f"\nResult saved to: {result_md_path}")


if __name__ == "__main__":
    main()