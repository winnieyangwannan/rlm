"""
Summarize MLE Bench rollout solutions using RLM.

This script:
1. Loads MLE Bench trajectory data from JSONL
2. Processes each rollout in parallel using RLM to generate solution summaries
3. Saves summaries as timestamped .md files with error detection and auto-retry

Features:
- Skips already processed rollouts (unless previous run had errors)
- Auto-deletes and reprocesses failed summaries
- Parallel processing with configurable worker count
"""

import argparse
import os
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from rlm import RLM
from rlm.logger import RLMLogger
import json
import base64

load_dotenv()


# =============================================================================
# Configuration (defaults - can be overridden via CLI)
# =============================================================================
DEFAULT_CONFIG = {
    "run_id": "514",
    "model_name": "gpt-5",
    "job_name": "summarization",
    "task_name": "tweet-sentiment-extraction",
    "log_dir": "/checkpoint/maui_sft/winnieyangwn/rlm_dumps/summarization/v5/",
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
    parser.add_argument("--task_name", type=str, default=DEFAULT_CONFIG["task_name"], help="Specific task name to analyze (optional)")
    parser.add_argument("--log_dir", type=str, default=DEFAULT_CONFIG["log_dir"], help="Directory for log files")
    parser.add_argument("--max_depth", type=int, default=2, help="Max recursion depth for RLM")
    parser.add_argument("--max_iterations", type=int, default=5, help="Max iterations for RLM")
    parser.add_argument("--max_workers", type=int, default=20, help="Max parallel workers for processing rollouts")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    return parser.parse_args()


def validate_path(path: str, description: str) -> Path:
    """Validate that a path exists and return Path object."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    return p


def get_data_path(account: str, run_id: str) -> str:
    """Get the data path for a given run ID."""
    return f"/checkpoint/{account}/winnieyangwn/amaia_dumps/{run_id}/trajectories/{run_id}_metadata.jsonl"


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
            # Check for various error indicators
            if not content:
                return True
            if content.startswith("Error:"):
                return True
            # Check for common REPL error patterns
            if "Traceback (most recent call last)" in content[:500]:
                return True
            return False
    except Exception:
        return True  # If we can't read the file, treat it as an error


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
DATA_SCHEMA = """
================================================================================
AVAILABLE VARIABLE: `rollout_row`
================================================================================
A dictionary containing a single MLE Bench rollout (Rollout #{rollout_idx}).
Score: percentile={percentile}, medal={medal}

DICT KEYS:
├── task_name: str          # Task ID
├── task_description: str   # Full task description (markdown)
├── code: str | None        # Final submitted Python solution
├── percentile: float | None  # Score 0-1 (higher = better)
├── medal: str | None       # "gold", "silver", "bronze", or ""
├── valid_submission: bool  # Did agent produce valid submission?
├── eval_error_output: str  # Success/error details
├── eval_duration: float    # GPU eval time (seconds)
├── rollout_duration: float # Total rollout time (seconds)
└── rollout: list[dict]     # Multi-turn interaction transcript
    ├── turn_id: int
    ├── action: str         # Agent's response
    └── observation: str    # Environment's response

ACCESS EXAMPLES:
  rollout_row["code"]              # The submitted code
  rollout_row["percentile"]        # Score
  len(rollout_row["rollout"])      # Number of turns
"""

SUMMARY_TEMPLATE = """## Task

Analyze the code solution for Rollout #{rollout_idx} in the Kaggle competition **{task_name}**.
Access the code via `rollout_row["code"]`.

---

## Solution Summary

**Rollout ID:** {rollout_idx}  
**Score Percentile:** {percentile}
**Medal Earned:** {medal}

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

## IMPORTANT: How to Your Final Answer

When you have completed your analysis:

1. **Store your complete final answer in a variable named exactly as `final_answer`**
2. **Before returning, verify the variable exists** by printing: `print("final_answer" in dir())`
3. **Return using exactly**: `FINAL_VAR(final_answer)`

Example pattern:
```python
# Build your final answer
final_answer = "Your complete analysis here..."

# Verify it exists before returning
print("Variable 'final_answer' exists:", "final_answer" in dir())
```

Then in your next response, use: FINAL_VAR(final_answer)"""


def build_data_schema_single_rollout(rollout_idx: int, percentile: float | None, medal: str | None) -> str:
    return DATA_SCHEMA.format(rollout_idx=rollout_idx, percentile=percentile, medal=medal)


def build_question_single_rollout(task_name: str, rollout_idx: int, percentile: float | None, medal: str | None) -> str:
    """Build the analysis question for a single rollout."""
    return SUMMARY_TEMPLATE.format(
        task_name=task_name,
        rollout_idx=rollout_idx,
        percentile=percentile,
        medal=medal or 'none'
    )


def process_single_rollout(
    args: argparse.Namespace,
    row: dict,
    rollout_idx: int,
    output_dir: Path,
) -> str | None:
    """Process a single rollout and return the summary.
    
    Args:
        args: Parsed command line arguments
        row: Dictionary containing the rollout data
        rollout_idx: Index/ID of this rollout
        output_dir: Directory to save output files
        
    Returns:
        The summary markdown string, or None if processing failed
    """
    
    task_name = row.get("task_name", "unknown")
    percentile = row.get("percentile")
    medal = row.get("medal")
    
    # Skip if no code
    if not row.get("code"):
        print(f"  Skipping rollout {rollout_idx}: no code submitted")
        return None
    
    print(f"\n{'='*60}")
    print(f"Processing Rollout {rollout_idx}: {task_name}")
    print(f"  Percentile: {percentile}, Medal: {medal}")
    print(f"{'='*60}")
    
    # Encode row as base64 JSON (avoids string escaping issues, no temp files)
    row_json_b64 = base64.b64encode(json.dumps(row).encode()).decode()
    
    # Set up logger for this rollout
    log_file_name = f"{args.model}_{args.job_name}_{args.run_id}_{task_name}_rollout{rollout_idx}"
    logger = RLMLogger(
        log_dir=str(output_dir),
        file_name=log_file_name
    )
    
    # Setup code: decode base64 JSON to load rollout data
    setup_code = textwrap.dedent(f"""
    import pandas as pd
    import json
    import base64

    # Load single rollout data from base64-encoded JSON
    rollout_row = json.loads(base64.b64decode('{row_json_b64}').decode())

    print(f"Loaded rollout #{rollout_idx} for task: {{rollout_row['task_name']}}")
    print(f"Percentile: {{rollout_row['percentile']}}, Medal: {{rollout_row['medal']}}")
    """)
    
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
    
    # Build prompt for single rollout
    data_schema = build_data_schema_single_rollout(rollout_idx, percentile, medal)
    question = build_question_single_rollout(task_name, rollout_idx, percentile, medal)
    root_prompt = f"{data_schema}\n\nQUESTION:\n{question}"
    
    # Run RLM completion
    print(f"Running RLM analysis (max_depth={args.max_depth}, max_iterations={args.max_iterations})...")
    result = rlm.completion(
        prompt="",
        root_prompt=root_prompt
    )
    
    # Save result with same name as log file but .md extension
    # logger.log_file_path includes timestamp and UUID, e.g. "...rollout728_2026-02-19_23-27-21_1f0bdf49.jsonl"
    output_path = Path(logger.log_file_path).with_suffix(".md")
    summary = save_summarization_from_result(result, str(output_path))
    
    return summary


def main() -> None:
    args = parse_args()
    
    # Build paths
    data_path = get_data_path(args.account, args.run_id)
    validate_path(data_path, "Data file")
    
    # Validate required environment variables
    required_env_vars = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    # Load data into Python (outside RLM) to iterate over rollouts
    print(f"Loading data from {data_path}...")
    all_rollouts_df = pd.read_json(data_path, lines=True)
    print(f"Loaded {len(all_rollouts_df)} total rollouts")
    
    # Filter by task_name if specified
    if args.task_name:
        all_rollouts_df = all_rollouts_df[all_rollouts_df['task_name'] == args.task_name]
        print(f"Filtered to {len(all_rollouts_df)} rollouts for task: {args.task_name}")
    
    # Filter to only valid submissions (preserves original indices)
    all_rollouts_df = all_rollouts_df[all_rollouts_df['valid_submission'] == True]
    print(f"Filtered to {len(all_rollouts_df)} valid submissions")
    
    if len(all_rollouts_df) == 0:
        print("No rollouts found matching criteria. Exiting.")
        return
    
    # Create output directory
    output_dir = Path(args.log_dir)
    if args.task_name:
        output_dir = output_dir / args.task_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Track results
    successful = []
    failed = []
    skipped = []
    
    # Build list of rollouts to process (skip already processed without errors)
    rollouts_to_process = []
    for df_idx, row in all_rollouts_df.iterrows():
        row_dict = row.to_dict()
        task_name = row_dict.get("task_name", "unknown")
        # Check for existing file with matching log_file_name pattern (includes timestamp/UUID)
        md_pattern = f"{args.model}_{args.job_name}_{args.run_id}_{task_name}_rollout{df_idx}_*.md"
        existing_md_files = list(output_dir.glob(md_pattern))
        if existing_md_files:
            # Check the most recent file for errors
            existing_md = max(existing_md_files, key=lambda p: p.stat().st_mtime)
            if has_error_in_md(existing_md):
                # Delete old .md and .jsonl files before reprocessing
                print(f"  Reprocessing rollout {df_idx}: previous run had error ({existing_md.name})")
                for md_file in existing_md_files:
                    md_file.unlink()
                    print(f"    Deleted: {md_file.name}")
                # Also delete corresponding .jsonl log file
                log_file_pattern = f"{args.model}_{args.job_name}_{args.run_id}_{task_name}_rollout{df_idx}_*.jsonl"
                for log_file in output_dir.glob(log_file_pattern):
                    log_file.unlink()
                    print(f"    Deleted: {log_file.name}")
                rollouts_to_process.append((df_idx, row_dict))
            else:
                print(f"  Skipping rollout {df_idx}: already processed ({existing_md.name})")
                skipped.append(df_idx)
        else:
            rollouts_to_process.append((df_idx, row_dict))
    
    # Process rollouts in parallel
    print(f"\nProcessing {len(rollouts_to_process)} rollouts with {args.max_workers} workers...")
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {}
        for df_idx, row_dict in rollouts_to_process:
            future = executor.submit(
                process_single_rollout,
                args,
                row_dict,
                df_idx,
                output_dir,
            )
            futures[future] = df_idx
        
        for future in as_completed(futures):
            df_idx = futures[future]
            try:
                summary = future.result()
                if summary:
                    successful.append(df_idx)
                    print(f"  Completed rollout {df_idx}")
                else:
                    skipped.append(df_idx)
            except Exception as e:
                print(f"  ERROR processing rollout {df_idx}: {e}")
                failed.append((df_idx, str(e)))
    
    # Summary
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"  Successful: {len(successful)}")
    print(f"  Skipped (invalid/no code/already processed): {len(skipped)}")
    print(f"  Failed: {len(failed)}")
    if failed:
        print(f"  Failed rollouts: {[f[0] for f in failed]}")


if __name__ == "__main__":
    main()
