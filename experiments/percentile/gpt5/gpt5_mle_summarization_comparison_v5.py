"""
Comparison pipeline for MLE Bench rollout summaries.

This script loads pre-generated rollout summaries (individual .md files per rollout)
and runs a comparison analysis using RLM to identify patterns across solutions.

Workflow:
1. Load all rollout summaries for a task from {summaries_dir}/{task_name}_rollout*_*.md
2. Build a DataFrame with columns: rollout_idx, percentile, medal, code_summary
3. Run comparison analysis using RLM with the DataFrame
"""

import argparse
import os
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from rlm import RLM
from rlm.logger import RLMLogger

load_dotenv()


# =============================================================================
# Configuration (defaults - can be overridden via CLI)
# =============================================================================
DEFAULT_CONFIG = {
    "run_id": "514",
    "model_name": "gpt-5",
    "job_name": "summarization_comparison",
    "log_dir": "/checkpoint/maui_sft/winnieyangwn/rlm_dumps/comparison/",
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare MLE Bench rollout summaries with RLM")
    parser.add_argument("--run-id", type=str, default=DEFAULT_CONFIG["run_id"], help="Run ID to analyze")
    parser.add_argument("--model", type=str, default=DEFAULT_CONFIG["model_name"], help="Model name to use")
    parser.add_argument("--job-name", type=str, default=DEFAULT_CONFIG["job_name"], help="Job name for logging")
    parser.add_argument("--task-name", type=str, required=True, help="Task name to analyze")
    parser.add_argument("--summaries-dir", type=str, required=True, help="Directory containing rollout summary .md files")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_CONFIG["log_dir"], help="Directory for output files")
    parser.add_argument("--max-depth", type=int, default=2, help="Max recursion depth for RLM")
    parser.add_argument("--max-iterations", type=int, default=20, help="Max iterations for RLM")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    return parser.parse_args()


def has_error_in_md(md_path: Path) -> bool:
    """Check if an existing .md file contains an error.
    
    Args:
        md_path: Path to the .md file
        
    Returns:
        True if file contains an error or is empty/unreadable, False otherwise
    """
    try:
        with open(md_path) as f:
            content = f.read().strip()
            if not content:
                return True
            if content.startswith("Error:"):
                return True
            if "Traceback (most recent call last)" in content[:500]:
                return True
        return False
    except Exception:
        return True


def parse_rollout_idx_from_filename(filename: str, task_name: str) -> int | None:
    """Extract rollout index from filename.
    
    Pattern: {task_name}_rollout{idx}_{timestamp}.md
    
    Args:
        filename: The filename to parse
        task_name: Task name prefix
        
    Returns:
        Rollout index as int, or None if parsing fails
    """
    # Pattern: task_name_rollout{idx}_{timestamp}.md
    pattern = rf"{re.escape(task_name)}_rollout(\d+)_.*\.md"
    match = re.match(pattern, filename)
    if match:
        return int(match.group(1))
    return None


def parse_metadata_from_summary(content: str) -> tuple[float | None, str | None]:
    """Extract percentile and medal from summary content.
    
    The summary format includes lines like:
    **Score Percentile:** 0.85
    **Medal Earned:** gold
    
    Args:
        content: The markdown summary content
        
    Returns:
        Tuple of (percentile, medal) or (None, None) if parsing fails
    """
    percentile = None
    medal = None
    
    # Parse percentile
    percentile_match = re.search(r"\*\*Score Percentile:\*\*\s*([\d.]+|None)", content)
    if percentile_match:
        val = percentile_match.group(1)
        if val != "None":
            try:
                percentile = float(val)
            except ValueError:
                pass
    
    # Parse medal
    medal_match = re.search(r"\*\*Medal Earned:\*\*\s*(\w+)", content)
    if medal_match:
        medal = medal_match.group(1).lower()
        if medal == "none":
            medal = None
    
    return percentile, medal


def load_summaries_to_df(summaries_dir: Path, task_name: str) -> pd.DataFrame | None:
    """Load rollout summaries into a DataFrame.
    
    Args:
        summaries_dir: Directory containing the .md summary files
        task_name: Task name to filter summaries
        
    Returns:
        DataFrame with columns: rollout_idx, percentile, medal, code_summary
        Or None if no valid summaries found
    """
    # Find all .md files for this task (pattern: {task_name}_rollout{idx}_{timestamp}.md)
    md_files = sorted(summaries_dir.glob(f"{task_name}_rollout*_*.md"))
    
    if not md_files:
        print(f"No summaries found for task: {task_name}")
        return None
    
    print(f"Found {len(md_files)} rollout summary files for task: {task_name}")
    
    rows = []
    for md_file in md_files:
        if has_error_in_md(md_file):
            print(f"  Skipping (error): {md_file.name}")
            continue
        
        content = md_file.read_text().strip()
        if not content:
            print(f"  Skipping (empty): {md_file.name}")
            continue
        
        # Parse rollout index from filename
        rollout_idx = parse_rollout_idx_from_filename(md_file.name, task_name)
        if rollout_idx is None:
            print(f"  Skipping (bad filename): {md_file.name}")
            continue
        
        # Parse percentile and medal from content
        percentile, medal = parse_metadata_from_summary(content)
        
        rows.append({
            "rollout_idx": rollout_idx,
            "percentile": percentile,
            "medal": medal,
            "code_summary": content,
        })
        print(f"  Loaded rollout {rollout_idx}: percentile={percentile}, medal={medal} ({len(content)} chars)")
    
    if not rows:
        print(f"No valid summaries found for task: {task_name}")
        return None
    
    df = pd.DataFrame(rows)
    df = df.sort_values("rollout_idx").reset_index(drop=True)
    print(f"Created DataFrame with {len(df)} rollouts")
    return df


def save_comparison_result(result, output_path: str) -> str | None:
    """Save RLM comparison result as .md file.
    
    Args:
        result: The RLMChatCompletion object or str from rlm.completion()
        output_path: Path to save the .md file
        
    Returns:
        The final answer string, or None if result was empty/error
    """
    if not result:
        print("No result to save")
        return None
    
    if isinstance(result, str):
        final_answer = result
    else:
        final_answer = result.response
    
    if not final_answer:
        print("No final answer in result")
        return None
    
    if final_answer.startswith("Error:"):
        print(f"REPL execution failed: {final_answer}")
        return None
    
    output_path_obj = Path(output_path)
    if output_path_obj.suffix != ".md":
        output_path_obj = output_path_obj.with_suffix(".md")
    
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path_obj, "w") as md_file:
        md_file.write(final_answer)
    print(f"Saved comparison result to {output_path_obj} ({len(final_answer)} chars)")
    
    return final_answer


# =============================================================================
# Schema and Prompt Templates
# =============================================================================

def build_data_schema(num_summaries: int, task_name: str) -> str:
    return f"""
================================================================================
AVAILABLE VARIABLES (pre-loaded in REPL namespace)
================================================================================
  - `task_summaries`: pandas DataFrame with {num_summaries} rollout summaries for task "{task_name}"
  - `task_name`: "{task_name}"
  - `pd`: pandas module

⚠️ WARNING: Do NOT call globals() or locals() - they are disabled.
⚠️ WARNING: Do NOT reassign these variables.

================================================================================
DATAFRAME SCHEMA: `task_summaries`
================================================================================
Columns:
├── rollout_idx: int       # Rollout index/ID
├── percentile: float      # Score 0-1 (higher = better), may be None
├── medal: str | None      # "gold", "silver", "bronze", or None
└── code_summary: str      # Full markdown summary of the solution

ACCESS EXAMPLES:
  task_summaries["rollout_idx"].tolist()           # List all rollout indices
  task_summaries["percentile"].describe()          # Summary stats of scores
  task_summaries.loc[0, "code_summary"]            # First rollout's summary
  task_summaries[task_summaries["percentile"] >= 0.8]  # High performers
  task_summaries[task_summaries["medal"].notna()]  # Medal winners
"""


# Import the contrastive template from v5 comparison
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

## REPL-First Workflow

**You are in a REPL environment.** You MUST execute code before producing any answer.

### Required Workflow:
1. **FIRST**: Run exploratory code (e.g., `print(task_summaries)`, `print(task_summaries.loc[0, "code_summary"][:2000])`)
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


def build_comparison_question(task_name: str) -> str:
    return f"""## Task

Analyze the pre-generated rollout summaries for **{task_name}** using the `task_summaries` DataFrame.

{CONTRASTIVE_REFLECTION_TEMPLATE}"""


def run_comparison(
    args: argparse.Namespace,
    task_summaries_df: pd.DataFrame,
) -> str | None:
    """Run the comparison analysis using RLM.
    
    Args:
        args: Parsed command line arguments
        task_summaries_df: DataFrame with rollout summaries
        
    Returns:
        The comparison result string, or None if failed
    """
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate required environment variables
    required_env_vars = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    # Save DataFrame to parquet file for REPL to load
    df_file = f"/tmp/task_summaries_{args.task_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
    task_summaries_df.to_parquet(df_file, index=False)
    print(f"Saved task_summaries DataFrame to: {df_file}")
    
    # Setup code: load DataFrame into REPL
    setup_code = f"""
import pandas as pd

# Load task summaries DataFrame
task_name = "{args.task_name}"
task_summaries = pd.read_parquet("{df_file}")
print(f"Loaded task_summaries for task: {{task_name}}")
print(f"  Shape: {{task_summaries.shape}}")
print(f"  Columns: {{task_summaries.columns.tolist()}}")
print(f"  Rollout indices: {{task_summaries['rollout_idx'].tolist()}}")
print(f"  Percentile range: {{task_summaries['percentile'].min():.3f}} - {{task_summaries['percentile'].max():.3f}}")
print(f"  Medal winners: {{task_summaries['medal'].notna().sum()}}")
"""
    
    # Set up logger
    log_file_name = f"{args.model}_{args.job_name}_{args.run_id}_{args.task_name}"
    logger = RLMLogger(
        log_dir=str(output_dir),
        file_name=log_file_name
    )
    
    # Create RLM instance
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
            "setup_code": setup_code,
        },
        max_depth=args.max_depth,
        max_iterations=args.max_iterations,
        logger=logger,
        verbose=args.verbose,
    )
    
    # Build prompt
    num_summaries = len(task_summaries_df)
    data_schema = build_data_schema(num_summaries, args.task_name)
    question = build_comparison_question(args.task_name)
    root_prompt = f"{data_schema}\n\nQUESTION:\n{question}"
    
    # Run RLM completion
    print(f"\nRunning comparison analysis (max_depth={args.max_depth}, max_iterations={args.max_iterations})...")
    result = rlm.completion(
        prompt="",
        root_prompt=root_prompt
    )
    
    # Save result
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = output_dir / f"{args.task_name}_comparison_{timestamp}.md"
    comparison_result = save_comparison_result(result, str(output_path))
    
    print(f"\nLog saved to: {logger.log_file_path}")
    
    return comparison_result


def main() -> None:
    args = parse_args()
    
    # Validate summaries directory
    summaries_dir = Path(args.summaries_dir)
    if not summaries_dir.exists():
        raise FileNotFoundError(f"Summaries directory not found: {summaries_dir}")
    
    print(f"\n{'='*80}")
    print(f"MLE Bench Summarization Comparison v5")
    print(f"{'='*80}")
    print(f"Task: {args.task_name}")
    print(f"Summaries dir: {summaries_dir}")
    print(f"Model: {args.model}")
    print(f"{'='*80}\n")
    
    # Load summaries into DataFrame
    task_summaries_df = load_summaries_to_df(summaries_dir, args.task_name)
    
    if task_summaries_df is None or len(task_summaries_df) == 0:
        print(f"ERROR: No valid summaries found for task: {args.task_name}")
        return
    
    if len(task_summaries_df) < 2:
        print(f"WARNING: Only {len(task_summaries_df)} summary found. Comparison requires at least 2 summaries.")
        return
    
    # Run comparison
    result = run_comparison(args, task_summaries_df)
    
    if result:
        print(f"\n{'='*80}")
        print("COMPARISON COMPLETE")
        print(f"{'='*80}\n")
    else:
        print(f"\n{'='*80}")
        print("COMPARISON FAILED")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
