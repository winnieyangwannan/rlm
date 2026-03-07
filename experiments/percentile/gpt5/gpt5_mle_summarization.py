"""
Quickstart example for analyzing MLE Bench rollout data with RLM.

This script demonstrates how to:
1. Load flattened MLE Bench trajectory data as a pandas DataFrame (FAST)
2. Provide a data schema description in the root_prompt
3. Query the RLM to analyze the rollout data

Performance optimization: Uses setup_code to load data directly into REPL,
bypassing JSON serialization of large context data.
"""

import argparse
import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv

from rlm import RLM
from rlm.logger import RLMLogger
import json

load_dotenv()


# =============================================================================
# Configuration (defaults - can be overridden via CLI)
# =============================================================================
DEFAULT_CONFIG = {
    "run_id": "513",
    "model_name": "gpt-5",
    "job_name": "summarization",
    "task_name": "plant-pathology-2021-fgvc8",
    "log_dir": "/checkpoint/maui_sft/winnieyangwn/rlm_dumps/summarization/",
    "codebase_extensions": [".py", ".md", ".yaml"],
    "account": "maui_sft"
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze MLE Bench rollout data with RLM")
    parser.add_argument("--account", type=str, default=DEFAULT_CONFIG["account"], help="Account name for data path")
    parser.add_argument("--run_id", type=str, default=DEFAULT_CONFIG["run_id"], help="Run ID to analyze")
    parser.add_argument("--model", type=str, default=DEFAULT_CONFIG["model_name"], help="Model name to use")
    parser.add_argument("--job_name", type=str, default=DEFAULT_CONFIG["job_name"], help="Job name for logging")
    parser.add_argument("--task_name", type=str, default="iwildcam-2019-fgvc6", help="Specific task name to analyze (optional)")
    parser.add_argument("--log_dir", type=str, default=DEFAULT_CONFIG["log_dir"], help="Directory for log files")
    parser.add_argument("--max_depth", type=int, default=2, help="Max recursion depth for RLM")
    parser.add_argument("--max_iterations", type=int, default=10, help="Max iterations for RLM")
    parser.add_argument("--verbose", action="store_true", default=True, help="Enable verbose output")
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


def get_data_path(account: str, run_id: int) -> str:
    """Get the data path for a given run ID."""
    return f"/checkpoint/{account}/winnieyangwn/amaia_dumps/{run_id}/trajectories/{run_id}_metadata.jsonl"


def save_summarization_from_log(log_path: str) -> str | None:
    """Extract final_answer from RLM log and save as .md file.
    
    Args:
        log_path: Path to the JSONL log file from Round 1
        
    Returns:
        The final_answer string (markdown or JSON), or None if not found
    """
    log_path = log_path.strip()  # Remove any leading/trailing whitespace
    log_path_obj = Path(log_path)
    if not log_path_obj.exists():
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
                
                # Save final_answer as .md file in the same directory
                md_path = log_path_obj.with_suffix(".md")
                with open(md_path, "w") as md_file:
                    md_file.write(final_answer)
                print(f"Saved final_answer to {md_path} ({len(final_answer)} chars)")
                
                return final_answer
    
    print("No final_answer found in log")
    return None


def save_summarization_from_result(result, output_path: str) -> str | None:
    """Save RLM completion result as .md file.
    
    Args:
        result: The RLMChatCompletion object or str from rlm.completion()
        output_path: Path to save the .md file (will add .md suffix if needed)
        
    Returns:
        The final answer string, or None if result was empty/error
    """
    if not result:
        print("No result to save")
        return None
    
    # Extract final answer - handle both RLMChatCompletion object and plain string
    if isinstance(result, str):
        final_answer = result
    else:
        # RLMChatCompletion object - final answer is in .response
        final_answer = result.response
    
    if not final_answer:
        print("No final answer in result")
        return None
    
    # Check for error messages that indicate REPL execution failed
    if final_answer.startswith("Error:"):
        print(f"REPL execution failed: {final_answer}")
        return None
    
    # Ensure .md extension
    output_path_obj = Path(output_path)
    if output_path_obj.suffix != ".md":
        output_path_obj = output_path_obj.with_suffix(".md")
    
    # Create parent directory if needed
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path_obj, "w") as md_file:
        md_file.write(final_answer)
    print(f"Saved final answer to {output_path_obj} ({len(final_answer)} chars)")
    
    return final_answer


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
    """Build the analysis question, optionally scoped to a specific task."""
    task_filter = f"for the Kaggle competition **{task_name}**" if task_name else "across all tasks in the dataset"
    
    return f"""## Task

Analyze code solutions {task_filter} by documenting and understanding what each solution implements.

**How to access the data:**
- Each row in `rollout_df` is agent's attempt at the task
- `rollout_df["code"]` contains the final submitted Python solution (may be None if no valid submission)
- `rollout_df["valid_submission"]` indicates if the submission was valid
- `rollout_df["percentile"]` is the score (0-1, higher is better)
**Analysis scope:** Only analyze rows where valid_submission == True

---

## Part 0: Task Analysis

The analysis begins with understanding the task context from `rollout_df["task_description"].iloc[0]`:

1. **Problem Type**: The type of machine learning problem (Classification/Regression/Object Detection/etc.)
2. **Domain**: The application domain (Healthcare/Finance/etc.)
3. **Input Format**: The nature of the input data (Images/Tabular/Text/etc.)
4. **Evaluation Metric**: The metric used for scoring and what it optimizes for
5. **Key Challenges**: The factors that make this task difficult

---

## Part 1: Individual Solution Summaries

For each **valid** solution (where `valid_submission == True`), analyze the code and document:

### Solution Summary Template

**Solution ID:** [identifier]  
**Score Percentile:** [percentile]

#### 1. Data Preprocessing
The data pipeline from raw input to model-ready format, including what was actually implemented:
- Data loading and splitting approaches
- Cleaning strategies (missing values, outliers, filtering)
- Transformations applied (scaling, encoding, type conversions)
- Data augmentation techniques (if any)
- Other preprocessing steps
 
#### 2. Feature Engineering
- New features that were created and their derivation
- Feature selection or dimensionality reduction methods
- Domain-specific transformations
- When no feature engineering was performed, this is noted explicitly

#### 3. Model Selection
- The primary algorithm(s) used (exact model class/function)
- Model hyperparameters (learning rate, depth, n_estimators, etc.)
- Ensemble architecture (if any): stacking, blending, voting, etc.
- Number of models in the ensemble (if applicable)
- Pretrained models: which ones and how they were used (feature extraction, fine-tuning, etc.)

#### 4. Training Methodology
- Hyperparameter selection method (if any)
- Training configuration (relevant parameters for the model type)
- Other important training details (early stopping, regularization, etc.)

#### 5. Evaluation & Submission
- Final prediction method (mean, median, weighted average, etc.)
- Post-processing of predictions

#### 6. Notable Implementation Details
- Computational considerations (GPU usage, runtime optimizations)
- Other unique approaches or novel techniques
- Other significant aspects of the solution's approach

---

## Part 2: Performance Analysis

After documenting all solutions, provide deeper analysis for each:

### For High-Scoring Solutions (percentile ≥ 0.7)

For each high-scoring solution, explain:
- **Why it works**: Which specific technical choices (from Part 1) likely contributed most to the high score? Cite evidence from the code.
- **Key strengths**: What does this solution do particularly well compared to lower-scoring ones?
- **Alignment with task**: How do the choices align with the task's evaluation metric and challenges identified in Part 0?

### For Low-Scoring Solutions (percentile < 0.4)

For each low-scoring solution, explain:
- **Likely failure points**: Identify only the components that plausibly caused the low score. Not every aspect needs to be blamed—focus on:
  - Critical omissions (e.g., no cross-validation, missing essential preprocessing)
  - Poor choices for this specific task (e.g., wrong model family, metric mismatch)
  - Implementation issues (e.g., data leakage, incorrect target encoding)
- **Evidence**: For each identified issue, cite specific code evidence or contrast with what high-scoring solutions did differently.
- **What would help**: Concrete changes that would likely improve the score.

### For Medium-Scoring Solutions (0.4 ≤ percentile < 0.7)

Briefly note:
- What's working reasonably well
- The most likely bottleneck preventing higher performance

---

## IMPORTANT: Returning Your Final Answer

When you have completed your analysis:

1. **Store your complete final answer in a variable named exactly `final_answer`**
2. **Before returning, verify the variable exists** by printing: `print("final_answer" in dir())`
3. **Return using exactly**: `FINAL_VAR(final_answer)`

⚠️ Do NOT use a different variable name like `cleaned_final_output`, `result`, or `output`.
⚠️ Do NOT call FINAL_VAR with a variable that doesn't exist - this will cause an error.

Example pattern:
```python
# Build your final answer
final_answer = "Your complete analysis here..."

# Verify it exists before returning
print("Variable 'final_answer' exists:", "final_answer" in dir())
```

Then in your next response, use: FINAL_VAR(final_answer)"""


def main() -> None:
    args = parse_args()
    
    # Build paths
    data_path = get_data_path(args.account, args.run_id)
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

    # Set up logger
    log_file_name = f"{args.model}_{args.job_name}_{args.run_id}"
    if args.task_name:
        log_file_name += f"_{args.task_name}"
    logger = RLMLogger(
        log_dir=args.log_dir,
        file_name=log_file_name
    )

    # Setup code: load data directly into REPL (bypasses JSON serialization)
    # Optionally filter by task_name if specified
    if args.task_name:
        setup_code = f"""
        import pandas as pd

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

        # Load rollout data as DataFrame
        rollout_df = pd.read_json('{data_path}', lines=True)
        print(f"Loaded {{len(rollout_df)}} rollouts")
        """

    # Validate required environment variables
    required_env_vars = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

    # Create the RLM Instance
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

    # Build the question and root_prompt
    question = build_question(args.task_name)
    root_prompt = f"{data_schema}\n\nQUESTION:\n{question}"

    # Run RLM completion
    print(f"\nRunning RLM analysis (max_depth={args.max_depth}, max_iterations={args.max_iterations})...")
    print("(GPT-5 API calls may take 1-5+ minutes per iteration - please wait...)\n")
    result = rlm.completion(
        prompt="",
        root_prompt=root_prompt
    )

    # Save result as .md file (use log path as base, but with .md extension)
    output_path = Path(logger.log_file_path).with_suffix(".md")
    save_summarization_from_result(result, str(output_path))


if __name__ == "__main__":
    main()
