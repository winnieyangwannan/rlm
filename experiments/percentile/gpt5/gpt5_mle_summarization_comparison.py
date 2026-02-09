"""
Two-round pipeline for analyzing MLE Bench rollout data with RLM.

This script demonstrates a two-round analysis pipeline:
1. Round 1: Analyze each rollout → output structured JSON via FINAL_VAR
2. Round 2: Load analysis from log → compare/aggregate solutions → final insights

Performance optimization: Uses setup_code to load data directly into REPL,
bypassing JSON serialization of large context data.
"""

import argparse
import json
import os
import subprocess
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
    "job_name": "summarization_comparison",
    "log_dir": "/checkpoint/maui_sft/winnieyangwn/rlm_dumps",
    "codebase_extensions": [".py", ".md", ".yaml"],
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze MLE Bench rollout data with RLM")
    parser.add_argument("--run-id", type=int, default=DEFAULT_CONFIG["run_id"], help="Run ID to analyze")
    parser.add_argument("--model", type=str, default=DEFAULT_CONFIG["model_name"], help="Model name to use")
    parser.add_argument("--job-name", type=str, default=DEFAULT_CONFIG["job_name"], help="Job name for logging")
    parser.add_argument("--task-name", type=str, default="vinbigdata-chest-xray-abnormalities-detection", help="Specific task name to analyze (optional)")
    parser.add_argument("--max-depth", type=int, default=2, help="Max recursion depth for RLM")
    parser.add_argument("--max-iterations", type=int, default=20, help="Max iterations for RLM")
    parser.add_argument("--verbose", action="store_true", default=True, help="Enable verbose output")
    parser.add_argument("--round1-log", type=str, default="/checkpoint/maui_sft/winnieyangwn/rlm_dumps/gpt-5_summarization_comparison_round1_513_2026-02-08_07-19-05_710f5dca.jsonl", help="Path to existing Round 1 log file. If provided, skips Round 1 and starts from Round 2.")
    parser.add_argument("--round1-only", action="store_true", default=False, help="Only run Round 1 (skip Round 2 comparison). Useful for batch processing.")
    return parser.parse_args()


def get_row_count(path: str, task_name: str | None = None) -> int:
    """Get number of rows in JSONL file, optionally filtered by task_name."""
    if task_name:
        # Use grep to count only rows matching the task_name
        result = subprocess.run(
            ["grep", "-c", f'"task_name":"{task_name}"', path],
            capture_output=True, text=True
        )
        # grep returns exit code 1 if no matches found
        if result.returncode == 1:
            return 0
        if result.returncode != 0:
            raise RuntimeError(f"Failed to count rows in {path}: {result.stderr}")
        return int(result.stdout.strip())
    else:
        result = subprocess.run(["wc", "-l", path], capture_output=True, text=True, check=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to count rows in {path}: {result.stderr}")
        return int(result.stdout.split()[0])


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


# =============================================================================
# Data Schema Description (for root_prompt)
# =============================================================================
def build_data_schema(num_rollouts: int) -> str:
    return f"""
================================================================================
AVAILABLE VARIABLES (top-level, use directly - do NOT reassign these!)
================================================================================
The following variables are pre-loaded in the REPL namespace. Use them directly:
  - `rollout_df` (pandas.DataFrame) - MLE Bench rollout data
  - `pd` (module) - pandas is already imported

⚠️ WARNING: Do NOT call globals() or locals() - they are disabled.
⚠️ WARNING: Do NOT reassign these variables (e.g., `rollout_df = ...`).
   Just use them directly: `rollout_df.head()`, etc.

================================================================================
ROLLOUT DATA: `rollout_df`
================================================================================
A pandas DataFrame with {num_rollouts} MLE Bench rollouts. Each rollout is an LLM agent's 
attempt to solve an ML task (from Kaggle competitions) through multi-turn interaction.

DATAFRAME COLUMNS:
├── task_name: str          # Task ID, e.g. "detecting-insults-in-social-commentary"
├── task_description: str   # Full task description (markdown)
├── code: str | None        # Final submitted Python solution
├── percentile: float | None  # Score 0-1 (higher = better, 1 = top)
├── valid_submission: bool  # Did agent produce valid submission?
├── eval_error_output: str  # Success/error details during evaluation
├── eval_duration: float    # GPU eval time (seconds)
├── rollout_duration: float # Total rollout time (seconds)
└── rollout: list[dict]     # Multi-turn interaction transcript (stored as Python list)
    ├── turn_id: int        # Turn number (0-indexed)
    ├── action: str         # Agent's response (reasoning + tool calls, e.g. bash commands)
    └── observation: str    # Environment's response to the action

ACCESS EXAMPLES:
  rollout_df["task_name"].iloc[0]                    # First rollout's task
  rollout_df["percentile"].iloc[0]                   # First rollout's score
  len(rollout_df["rollout"].iloc[0])                 # Number of turns in first rollout
  rollout_df["rollout"].iloc[0][0]["action"]         # First action of the first rollout
  rollout_df.groupby("task_name")["percentile"].mean()  # Avg score by task
"""


def build_question(task_name: str | None = None) -> str:
    """Build the Round 1 analysis question - simple markdown output like gpt5_mle_summarization.py."""
    task_filter = f"for the Kaggle competition **{task_name}**" if task_name else "across all tasks in the dataset"
    
    return f"""## Task

Analyze code solutions {task_filter} by documenting what each solution implements.

**How to access the data:**
- Each row in `rollout_df` is one agent's attempt at the task
- `rollout_df["code"]` contains the final submitted Python solution (may be None if no valid submission)
- `rollout_df["valid_submission"]` indicates if the submission was valid
- `rollout_df["percentile"]` is the score (0-1, higher is better)

---

## Part 0: Task Analysis

Extract from `rollout_df["task_description"].iloc[0]`:

1. **Problem Type**: Classification/Regression/Object Detection/etc.
2. **Domain**: Healthcare/Finance/etc.
3. **Input Format**: Images/Tabular/Text/etc.
4. **Evaluation Metric**: What metric is used and what it optimizes for
5. **Key Challenges**: What makes this task difficult?

---

## Part 1: Individual Solution Summaries

For each **valid** solution (where `valid_submission == True`), analyze the code and document:

### Solution Summary Template

**Solution ID:** [identifier]  
**Score Percentile:** [percentile]

#### 1. Data Preprocessing
- Input data loading method
- Missing value handling (method, columns affected)
- Data cleaning steps (outlier removal, filtering, etc.)
- Normalization/scaling (which columns, which method)
- Train/test split approach

#### 2. Feature Engineering  
- Features created (list each with formula/method if possible)
- Feature selection/reduction techniques used
- Domain-specific transformations

#### 3. Synthetic Data / Data Augmentation
- Whether synthetic data was generated: Yes/No
- If yes: Generation method, volume, and integration approach

#### 4. Model Selection
- Primary algorithm(s) used (exact model class/function)
- Model hyperparameters (learning rate, depth, n_estimators, etc.)
- Ensemble approach (if any): stacking, blending, voting, etc.
- Pretrained models: [Which models, from where]

#### 5. Training Methodology
- Cross-validation scheme (k-fold, stratified, etc.)
- Hyperparameter tuning approach
- Early stopping criteria (if applicable)

#### 6. Notable Implementation Details
- Any unique approaches or novel techniques
- Computational considerations (GPU usage, runtime optimizations)

---

## IMPORTANT: Returning Your Final Answer

When you have completed your analysis:

1. **Store your complete final answer in a variable named exactly `final_answer`**
2. **Before returning, verify the variable exists** by printing: `print("final_answer" in dir())`
3. **Return using exactly**: `FINAL_VAR(final_answer)`

⚠️ Do NOT use a different variable name.
⚠️ Do NOT call FINAL_VAR with a variable that doesn't exist.

Example pattern:
```python
# Build your final answer
final_answer = "Your complete analysis here..."

# Verify it exists before returning
print("Variable 'final_answer' exists:", "final_answer" in dir())
```

Then in your next response, use: FINAL_VAR(final_answer)"""


def build_round2_question() -> str:
    """Build the Round 2 comparison question."""
    return """## Task

You have access to `round1_analysis` - a text analysis of solutions from Round 1.
Compare the solutions and identify patterns that distinguish high vs low performers.

**Available Variables:**
- `round1_analysis`: String containing the full analysis from Round 1 (markdown format)
- `rollout_df`: Original rollout data (for additional context if needed)

---

## IMPORTANT: You MUST use the REPL first!

Before producing any answer, you MUST:
1. First run code to inspect `round1_analysis` (e.g., `print(round1_analysis[:5000])`)
2. Use `llm_query()` if needed to analyze the content
3. Build your analysis step by step using print statements
4. Only call FINAL_VAR when you have a complete answer in a variable

⚠️ Do NOT call FINAL() or FINAL_VAR() without first running REPL code!
⚠️ Your first action should ALWAYS be to run code that explores the data.

---

### A. Solution Classification

First, identify and categorize all solutions by performance:

**High Score Solutions (percentile >= 0.6):**
- List each with Solution ID and Score

**Low Score Solutions (percentile < 0.6):**
- List each with Solution ID and Score

### B. Pattern Matrix

Create a table comparing key implementation choices:

| Dimension | High Score Implementations | Low Score Implementations |
|-----------|---------------------------|---------------------------|
| Data preprocessing | [Methods used] | [Methods used] |
| Feature engineering | ... | ... |
| Data augmentation | ... | ... |
| Model selection | ... | ... |
| Training methodology | ... | ... |
| Notable details | ... | ... |

### C. Critical Differences

For each dimension where high and low scores diverge significantly:

**[Dimension name]**
- **What high-score solutions did:** [Description with frequency]
- **What low-score solutions did:** [Description with frequency]
- **Concrete difference:** [Specific technical difference]

### D. Key Insights

- **High-score convergence:** Which techniques appeared in most high-score solutions?
- **Low-score anti-patterns:** Which mistakes appeared in most low-score solutions?
- **Recommendations:** Based on patterns, what should future solutions prioritize?

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

    # Get row count without loading data (fast)
    print(f"Counting rows in {data_path}...")
    num_rollouts = get_row_count(data_path, args.task_name)
    if args.task_name:
        print(f"Found {num_rollouts} rollouts for task: {args.task_name}")
    else:
        print(f"Found {num_rollouts} rollouts")

    # Build schema description
    data_schema = build_data_schema(num_rollouts)

    # Validate required environment variables
    required_env_vars = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

    # Setup code: load data directly into REPL (bypasses JSON serialization)
    if args.task_name:
        setup_code = f"""
import pandas as pd
import json

# Load rollout data as DataFrame
rollout_df = pd.read_json('{data_path}', lines=True)
print(f"Loaded {{len(rollout_df)}} total rollouts")

# Filter to specific task
rollout_df = rollout_df[rollout_df['task_name'] == '{args.task_name}']
print(f"Filtered to {{len(rollout_df)}} rollouts for task: {args.task_name}")
"""
    else:
        setup_code = f"""
import pandas as pd
import json

# Load rollout data as DataFrame
rollout_df = pd.read_json('{data_path}', lines=True)
print(f"Loaded {{len(rollout_df)}} rollouts")
"""

    # =========================================================================
    # ROUND 1: Analyze each rollout → structured JSON (skip if --round1-log provided)
    # =========================================================================
    round1_log_path = None
    
    if args.round1_log:
        # Skip Round 1, use existing log
        print("\n" + "=" * 80)
        print(f"SKIPPING ROUND 1: Loading analysis from existing log")
        print(f"Log path: {args.round1_log}")
        print("=" * 80)
        
        round1_log_path = args.round1_log
        if not Path(round1_log_path).exists():
            raise FileNotFoundError(f"Round 1 log not found: {round1_log_path}")
    else:
        # Run Round 1
        print("\n" + "=" * 80)
        print("ROUND 1: Analyzing individual rollouts...")
        print("=" * 80)

        logger_round1 = RLMLogger(
            log_dir=DEFAULT_CONFIG["log_dir"],
            file_name=f"{args.model}_{args.job_name}_round1_{args.run_id}"
        )

        rlm_round1 = RLM(
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
            logger=logger_round1,
            verbose=args.verbose,
        )

        question_round1 = build_question(args.task_name)
        root_prompt_round1 = f"{data_schema}\n\nQUESTION:\n{question_round1}"

        print(f"\nRunning Round 1 (max_depth={args.max_depth}, max_iterations={args.max_iterations})...\n")
        result_round1 = rlm_round1.completion(
            prompt="",
            root_prompt=root_prompt_round1
        )

        print("\n" + "-" * 40)
        print("ROUND 1 COMPLETE")
        print(f"Log saved to: {logger_round1.log_file_path}")
        print("-" * 40)
        
        round1_log_path = logger_round1.log_file_path

    # =========================================================================
    # PARSE ROUND 1 LOG → extract final_answer
    # =========================================================================
    print("\nParsing Round 1 log for analysis...")
    round1_analysis = load_analysis_from_log(round1_log_path)

    if round1_analysis is None:
        print("ERROR: Could not extract analysis from Round 1 log. Exiting.")
        print(f"Log path: {round1_log_path}")
        return

    # =========================================================================
    # ROUND 2: Compare solutions using Round 1 analysis (skip if --round1-only)
    # =========================================================================
    if args.round1_only:
        print("\n" + "=" * 80)
        print("ROUND 1 ONLY MODE: Skipping Round 2 comparison")
        print(f"Round 1 log saved to: {round1_log_path}")
        print("=" * 80)
        return

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
        file_name=f"{args.model}_{args.job_name}_round2_{args.run_id}"
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
    - Each solution has: ID, percentile, preprocessing, feature engineering, 
      augmentation, model selection, training, notable details
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

    print("\n" + "=" * 80)
    print("ROUND 2 COMPLETE - FINAL COMPARISON RESULT:")
    print("=" * 80)
    print(result_round2)
    print("\n" + "-" * 40)
    print(f"Round 1 log: {round1_log_path}")
    print(f"Round 2 log: {logger_round2.log_file_path}")
    print(f"Analysis file: {analysis_file}")
    print("-" * 40)


if __name__ == "__main__":
    main()
