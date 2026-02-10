"""
Round 2 comparison pipeline for analyzing MLE Bench rollout data with RLM.

This script performs comparison analysis given an existing Round 1 summarization log:
- Load analysis from Round 1 log → compare/aggregate solutions → final insights

Requires: A completed Round 1 summarization log (from gpt5_mle_summarization.py or similar)
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
    "log_dir": "/checkpoint/maui_sft/winnieyangwn/rlm_dumps",
    "codebase_extensions": [".py", ".md", ".yaml"],
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare MLE Bench solutions given Round 1 summarization log")
    parser.add_argument("--round1-log", type=str, default="/checkpoint/maui_sft/winnieyangwn/rlm_dumps/archive/gpt5/summarization/gpt-5_summarization_513_2026-02-08_23-35-35_c4318c42.jsonl", help="Path to Round 1 summarization log file (required)")
    parser.add_argument("--run-id", type=int, default=DEFAULT_CONFIG["run_id"], help="Run ID to analyze")
    parser.add_argument("--model", type=str, default=DEFAULT_CONFIG["model_name"], help="Model name to use")
    parser.add_argument("--job-name", type=str, default=DEFAULT_CONFIG["job_name"], help="Job name for logging")
    parser.add_argument("--task-name", type=str, default=None, help="Specific task name to analyze (optional)")
    parser.add_argument("--max-depth", type=int, default=2, help="Max recursion depth for RLM")
    parser.add_argument("--max-iterations", type=int, default=20, help="Max iterations for RLM")
    parser.add_argument("--verbose", action="store_true", default=True, help="Enable verbose output")
    return parser.parse_args()


def validate_path(path: str, description: str) -> Path:
    """Validate that a path exists and return Path object."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    return p


def get_data_path(run_id: int) -> str:
    """Get the data path for a given run ID."""
    return f"/checkpoint/maui_sft/winnieyangwn/amaia_dumps/{run_id}/trajectories/{run_id}_metadata.jsonl"


def load_analysis_from_log(log_path: str) -> str | None:
    """Extract final_answer from RLM log.
    
    Args:
        log_path: Path to the JSONL log file from Round 1
        
    Returns:
        The final_answer string (markdown or JSON), or None if not found
    """
    log_path = log_path.strip()  # Remove any leading/trailing whitespace
    if not Path(log_path).exists():
        print(f"Log file not found: {log_path}")
        return None
    
    with open(log_path) as f:
        for line in f:
            entry = json.loads(line)
            # Find the iteration with a final_answer
            if entry.get("type") == "iteration" and entry.get("final_answer"):
                final_answer = entry["final_answer"]
                # Check for error messages that indicate REPL execution failed
                if final_answer.startswith("Error:"):
                    print(f"REPL execution failed: {final_answer}")
                    return None
                print(f"Loaded final_answer from log ({len(final_answer)} chars)")
                return final_answer
    
    print("No final_answer found in log")
    return None


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
    return """## Goal
Analyze previous solutions and analysis to identify:
1. **Patterns that distinguish high vs low performance** (optimization)
2. **Critical failures that cause suboptimal results** (failure prevention)
3. **Unique high-performing approaches** (diversity preservation)

---

## Available Variables:**
- `round1_analysis`: String containing the full analysis from Round 1 (markdown format)
- `rollout_df`: Original rollout data (for additional context if needed)

---

## REPL-First Workflow

**You are in a REPL environment.** You MUST execute code before producing any answer.

### Required Workflow:
1. **FIRST**: Run exploratory code (e.g., `print(round1_analysis[:3000])`)
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

## Analysis Structure

### 1. Contrastive Pattern Analysis (High vs Low)

Use these thresholds throughout your analysis:
- **High Score Solutions**: percentile >= 0.6
- **Low Score Solutions**: percentile < 0.6

For each key dimension (Data preprocessing, feature engineering, model selection, training methodology, Evaluation & Submission,  Notable Implementation Details):

**[Dimension Name]**
- **What high-score solutions did**: [Description with frequency/count]
- **What low-score solutions did**: [Description with frequency/count]  
- **Concrete difference**: [Specific technical difference that explains the performance gap]
- **Why do these patterns matter?** (Underlying reasoning)
- **What should future solvers do?** (Actionable recommendations)


### 2. Unique High-Performing Approaches (Diversity Preservation)

**GOAL: Identify rare but effective techniques that could be lost in convergence to common patterns.**

Examine **top-tier solutions (≥ 0.8 percentile)** for techniques that are:
- ✅ **Rare**: low frequency (not common among high score solutions)
- ✅ **Effective**: unique and effective techniques that could contribute to the success of top-scoring solutions

---

## Returning Your Final Answer

After you have analyzed the data using REPL code:
1. Store your complete answer in a variable named `final_answer`
2. Call: `FINAL_VAR(final_answer)`"""


def main() -> None:
    args = parse_args()
    
    # Build paths
    data_path = get_data_path(args.run_id)
    validate_path(data_path, "Data file")
    
    # Validate Round 1 log exists
    round1_log_path = args.round1_log
    if not Path(round1_log_path).exists():
        raise FileNotFoundError(f"Round 1 log not found: {round1_log_path}")

    # Validate required environment variables
    required_env_vars = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

    # =========================================================================
    # PARSE ROUND 1 LOG → extract final_answer
    # =========================================================================
    print("\n" + "=" * 80)
    print(f"Loading analysis from Round 1 log: {round1_log_path}")
    print("=" * 80)
    
    round1_analysis = load_analysis_from_log(round1_log_path)

    if round1_analysis is None:
        print("ERROR: Could not extract analysis from Round 1 log. Exiting.")
        print(f"Log path: {round1_log_path}")
        return

    # =========================================================================
    # ROUND 2: Compare solutions using Round 1 analysis
    # =========================================================================
    print("\n" + "=" * 80)
    print("ROUND 2: Comparing solutions...")
    print("=" * 80)

    # Save Round 1 analysis to file for Round 2 to load
    analysis_file = f"/tmp/round1_analysis_{args.run_id}_{args.task_name or 'all'}.txt"
    prepare_round2_context(round1_analysis, analysis_file)

    setup_code_round2 = f"""
import pandas as pd

# Load original rollout data
rollout_df = pd.read_json('{data_path}', lines=True)
{"rollout_df = rollout_df[rollout_df['task_name'] == '" + args.task_name + "']" if args.task_name else ""}
print(f"Loaded {{len(rollout_df)}} rollouts")

# Load Round 1 analysis (markdown text)
with open('{analysis_file}') as f:
    round1_analysis = f.read()
print(f"Loaded Round 1 analysis ({{len(round1_analysis)}} chars)")
"""

    logger_round2 = RLMLogger(
        log_dir=DEFAULT_CONFIG["log_dir"],
        file_name=f"{args.model}_{args.job_name}_{args.run_id}"
    )

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
            "setup_code": setup_code_round2,
        },
        max_depth=args.max_depth,
        max_iterations=args.max_iterations,
        logger=logger_round2,
        verbose=args.verbose,
    )

    # Build Round 2 prompt
    round2_schema = f"""
================================================================================
AVAILABLE VARIABLES
================================================================================
  - `round1_analysis`: String containing the full analysis from Round 1 (markdown format)
    - Contains task analysis and individual solution summaries
    - Each solution has: ID, percentile, data preprocessing, feature engineering, 
      model selection, training methodology, evaluation & submission, notable details
  - `rollout_df`: Original rollout DataFrame (for additional context if needed)
  - `pd`: pandas is already imported
"""
    question_round2 = build_round2_question()
    root_prompt_round2 = f"{round2_schema}\n\nQUESTION:\n{question_round2}"

    print(f"\nRunning Round 2 (max_depth={args.max_depth}, max_iterations={args.max_iterations})...\n")
    result_round2 = rlm_round2.completion(
        prompt="",
        root_prompt=root_prompt_round2
    )

    # Save result to .md file with same name and location as log file
    log_path = Path(logger_round2.log_file_path)
    result_md_path = log_path.with_suffix(".md")
    with open(result_md_path, "w") as f:
        f.write(result_round2)
    print(f"\nResult saved to: {result_md_path}")


if __name__ == "__main__":
    main()