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

load_dotenv()


# =============================================================================
# Configuration (defaults - can be overridden via CLI)
# =============================================================================
DEFAULT_CONFIG = {
    "run-id": "520-rlm-comparison-argparse-xray",
    "model-name": "gpt-5",
    "job-name": "post-improvement_summarization",
    "log-dir": "/checkpoint/maui_sft/winnieyangwn/rlm_dumps/post-improvement/",
    "codebase-extensions": [".py", ".md", ".yaml"],
    "account": "agentic-models"
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze MLE Bench rollout data with RLM")
    parser.add_argument("--account", type=str, default=DEFAULT_CONFIG["account"], help="Account name for data path")
    parser.add_argument("--run-id", type=str, default=DEFAULT_CONFIG["run-id"], help="Run ID to analyze")
    parser.add_argument("--model", type=str, default=DEFAULT_CONFIG["model-name"], help="Model name to use")
    parser.add_argument("--job-name", type=str, default=DEFAULT_CONFIG["job-name"], help="Job name for logging")
    parser.add_argument("--task-name", type=str, default="iwildcam-2019-fgvc6", help="Specific task name to analyze (optional)")
    parser.add_argument("--log-dir", type=str, default=DEFAULT_CONFIG["log-dir"], help="Directory for log output")
    parser.add_argument("--max-depth", type=int, default=2, help="Max recursion depth for RLM")
    parser.add_argument("--max-iterations", type=int, default=10, help="Max iterations for RLM")
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
LLM HELPER FUNCTIONS (for sub-analysis)
================================================================================
  - `llm_query(prompt)` - Call the LLM with a single prompt (SLOW - sequential)
  - `llm_query_batched(prompts)` - Call the LLM with a list of prompts in PARALLEL (FAST)

⚡ PERFORMANCE TIP: When analyzing multiple items (e.g., summarizing 64 code solutions),
   ALWAYS use `llm_query_batched()` to process them in parallel. Example:
   
   ```python
   # SLOW - sequential calls (DON'T DO THIS):
   results = [llm_query(f"Analyze: {{code}}") for code in code_list]
   
   # FAST - parallel calls (DO THIS INSTEAD):
   prompts = [f"Analyze: {{code}}" for code in code_list]
   results = llm_query_batched(prompts)  # All calls run concurrently!
   ```

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

## Output Format: Structured JSON

For each **valid** solution (where `valid_submission == True`), analyze the code and return a JSON array of solution summaries.

### Required JSON Schema

```json
[
  {{{{
    "solution_id": "Unique identifier for this solution (e.g., row index)",
    "score_percentile": "Float between 0-1, or null if not available",
    "data_preprocessing": "Describe the data pipeline from raw input to model-ready format: data loading and splitting approaches, cleaning strategies (missing values, outliers, filtering), transformations applied (scaling, encoding, type conversions), data augmentation techniques (if any), and other preprocessing steps",
    "feature_engineering": "Describe new features created and their derivation, feature selection or dimensionality reduction methods, domain-specific transformations. State explicitly if no feature engineering was performed",
    "model_selection": "Describe the primary algorithm(s) used (exact model class/function), model hyperparameters (learning rate, depth, n_estimators, etc.), ensemble architecture if any (stacking, blending, voting), number of models in ensemble if applicable, pretrained models and how they were used (feature extraction, fine-tuning, etc.)",
    "training_methodology": "Describe hyperparameter selection method (if any), training configuration (relevant parameters for the model type), other important training details (early stopping, regularization, etc.)",
    "evaluation_and_submission": "Describe final prediction method (mean, median, weighted average, etc.) and post-processing of predictions",
    "notable_implementation_details": "Describe computational considerations (GPU usage, runtime optimizations), unique approaches or novel techniques, and other significant aspects of the solution's approach"
  }}}}
]
```

### Field Instructions

- **solution_id**: String - unique identifier for the solution
- **score_percentile**: Number or null - the percentile score from the data
- **data_preprocessing**: String - comprehensive description of all data preprocessing steps
- **feature_engineering**: String - describe all feature engineering or explicitly state "No feature engineering performed"
- **model_selection**: String - full details of model architecture and configuration
- **training_methodology**: String - training approach and configuration details
- **evaluation_and_submission**: String - how predictions were generated and submitted
- **notable_implementation_details**: String - any unique or notable aspects of the implementation


---

## IMPORTANT: Returning Your Final Answer

When you have completed your analysis:

1. **Store the JSON array (as a Python list of dicts) in a variable named exactly `final_answer`**
2. **Before returning, verify the variable exists** by printing: `print("final_answer" in dir())`
3. **Return using exactly**: `FINAL_VAR(final_answer)`

⚠️ Do NOT use a different variable name like `cleaned_final_output`, `result`, or `output`.
⚠️ Do NOT call FINAL_VAR with a variable that doesn't exist - this will cause an error.

Example pattern:
```python
import json

# Build your final answer as a list of dictionaries
final_answer = [
    {{{{
        "solution_id": "solution_0",
        "score_percentile": 0.85,
        "data_preprocessing": "Loaded train.csv and test.csv using pandas...",
        "feature_engineering": "No feature engineering performed",
        "model_selection": "Used RandomForestClassifier with n_estimators=100...",
        "training_methodology": "5-fold cross-validation with default parameters...",
        "evaluation_and_submission": "Predictions averaged across folds...",
        "notable_implementation_details": "Used joblib for parallel processing..."
    }}}}
]

# Verify it exists before returning
print("Variable 'final_answer' exists:", "final_answer" in dir())
print(f"Number of solutions analyzed: {{len(final_answer)}}")
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

    # Extract and save summarization results
    import ast
    import json

    # Extract the response string from RLMChatCompletion object
    response_str = result.response

    # Parse final_answer string to list of dicts
    # Use ast.literal_eval since it uses single quotes (Python format, not JSON)
    try:
        data = ast.literal_eval(response_str)
    except (ValueError, SyntaxError) as e:
        print(f"Warning: Could not parse result as Python literal: {e}")
        print("Attempting JSON parse...")
        try:
            data = json.loads(response_str)
        except json.JSONDecodeError as e2:
            print(f"Error: Could not parse result as JSON either: {e2}")
            print(f"Raw result:\n{response_str}")
            return

    # Get the log path from logger
    log_path = logger.log_file_path

    # Simplify: just append "_extracted" to the original log filename
    log_path_obj = Path(log_path)
    output_filename = f"{log_path_obj.stem}_extracted.jsonl"
    output_path = log_path_obj.parent / output_filename

    # Save as JSONL (one JSON object per line)
    with open(output_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

    print(f"\nSaved {len(data)} items to: {output_path}")


if __name__ == "__main__":
    main()
