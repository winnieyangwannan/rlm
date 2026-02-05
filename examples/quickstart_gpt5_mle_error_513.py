"""
Quickstart example for analyzing MLE Bench rollout data with RLM.

This script demonstrates how to:
1. Load flattened MLE Bench trajectory data as a pandas DataFrame (FAST)
2. Provide a data schema description in the root_prompt
3. Query the RLM to analyze the rollout data

Performance optimization: Uses setup_code to load data directly into REPL,
bypassing JSON serialization of large context data.
"""

import os
import subprocess

from dotenv import load_dotenv

from rlm import RLM
from rlm.logger import RLMLogger

load_dotenv()

# =============================================================================
# Configuration
# =============================================================================
run_id = 513
DATA_PATH = f"/checkpoint/maui_sft/winnieyangwn/amaia_dumps/{run_id}/trajectories/{run_id}_metadata.jsonl"
model_name = "gpt-5"  # Example model name
job_name = "common_invalid_errors"
log_dir = "/checkpoint/maui_sft/winnieyangwn/rlm_dumps"


def get_row_count(path: str) -> int:
    """Get number of rows in JSONL file without loading it."""
    result = subprocess.run(["wc", "-l", path], capture_output=True, text=True)
    return int(result.stdout.split()[0])


# =============================================================================
# Data Schema Description (for root_prompt)
# =============================================================================
def build_data_schema(num_rollouts: int) -> str:
    return f"""
CONTEXT: A pandas DataFrame `context` with {num_rollouts} MLE Bench rollouts (LLM agent attempts to solve ML tasks from Kaggle competition through multi-turn interaction).

DATAFRAME COLUMNS:
├── task_name: str          # Task ID, e.g. "detecting-insults-in-social-commentary"
├── task_description: str   # Full task description (markdown)
├── code: str | None        # Final submitted Python solution
├── percentile: float | None  # Score 0-1 (higher = better, 1 = top)
├── valid_submission: bool  # Did agent produce valid submission?
├── eval_error_output: str # Success/error details during evaluation
├── eval_duration: float    # GPU eval time (seconds)
├── rollout_duration: float # Total rollout time (seconds)
└── rollout: list[dict]     # Multi-turn interaction transcript (stored as Python list)
    ├── turn_id: int        # Turn number (0-indexed)
    ├── action: str         # Agent's response (reasoning + tool calls, e.g. bash commands)
    └── observation: str    # Environment's response to the action

ACCESS EXAMPLES (use pandas operations for speed):
  context["task_name"].iloc[0]                    # First rollout's task
  context["percentile"].iloc[0]                   # First rollout's score
  len(context["rollout"].iloc[0])                 # Number of turns in first rollout
  context["rollout"].iloc[0][0]["action"]         # First action of the first rollout
  context.groupby("task_name")["percentile"].mean()  # Avg score by task
"""


def main():
    # Get row count without loading data (fast)
    print(f"Counting rows in {DATA_PATH}...")
    num_rollouts = get_row_count(DATA_PATH)
    print(f"Found {num_rollouts} rollouts")

    # Build schema description
    data_schema = build_data_schema(num_rollouts)

    # Set up logger
    logger = RLMLogger(log_dir=log_dir, file_name=f"{model_name}_{job_name}_{run_id}")

    # Setup code: load data directly into REPL as pandas DataFrame (bypasses JSON serialization)
    setup_code = f"""
import pandas as pd
context = pd.read_json('{DATA_PATH}', lines=True)
"""

    # Create the RLM Instance
    rlm = RLM(
        backend="azure_openai",
        backend_kwargs={
            "model_name": model_name,
            "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
            "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
            "azure_deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            "api_version": "2025-03-01-preview",
        },
        environment="local",
        environment_kwargs={
            "setup_code": setup_code,  # Load data directly in REPL
        },
        max_depth=1,
        max_iterations=10,  # Reduced from 30 - simple analytical task
        logger=logger,
        verbose=True,
    )

    # Define your question
    # question = "What percentage of rollouts produced a valid submission?"
    # question = "Among all invalid submissions, what are the top 5 most common evaluation error messages?"
    question = "Among all invalid submissions, what are the top 5 most common evaluation error messages?"

    # Build the root_prompt with data schema + question
    root_prompt = f"{data_schema}\nQUESTION: {question}"

    # Run RLM completion
    print(f"\nRunning RLM with question: {question}\n")
    result = rlm.completion(
        prompt="",                # Empty - data loaded via setup_code
        root_prompt=root_prompt   # Schema + question shown at each iteration
    )

    print("\n" + "=" * 80)
    print("FINAL RESULT:")
    print("=" * 80)
    print(result)


if __name__ == "__main__":
    main()
