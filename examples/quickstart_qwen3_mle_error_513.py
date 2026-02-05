"""
Quickstart example for analyzing MLE Bench rollout data with RLM.

This script demonstrates how to:
1. Load flattened MLE Bench trajectory data
2. Provide a data schema description in the root_prompt
3. Query the RLM to analyze the rollout data
"""

import json
import os

from dotenv import load_dotenv

from rlm import RLM
from rlm.logger import RLMLogger

load_dotenv()

# =============================================================================
# Data Schema Description (for root_prompt)
# =============================================================================
def build_data_schema(num_rollouts: int) -> str:
    return f"""
CONTEXT: {num_rollouts} MLE Bench rollouts (LLM agent attempts to solve ML tasks from Kaggle competition through multi-turn interaction).

SCHEMA (each rollout is a dict):
├── task_name: str          # Task ID, e.g. "detecting-insults-in-social-commentary"
├── task_description: str   # Full task description (markdown)
├── code: str | None        # Final submitted Python solution
├── percentile: float | None  # Score 0-1 (higher = better, 1 = top)
├── valid_submission: bool  # Did agent produce valid submission?
├── eval_error_output: str # Success/error details during evaluation
├── eval_duration: float    # GPU eval time (seconds)
├── rollout_duration: float # Total rollout time (seconds)
└── rollout: list[dict]     # Multi-turn interaction transcript
    ├── turn_id: int        # Turn number (0-indexed)
    ├── action: str         # Agent's response (reasoning + tool calls, e.g. bash commands)
    └── observation: str    # Environment's response to the action

ACCESS EXAMPLES:
  context[0]["task_name"]           # First rollout's task
  context[0]["percentile"]          # First rollout's score
  len(context[0]["rollout"])        # Number of turns
  context[0]["rollout"][0]["action"] # First action of the first rollout
  [r for r in context if r["valid_submission"]]  # Valid submissions
  [r for r in context if r["percentile"] and r["percentile"] >= 50]  # High performers
"""

# =============================================================================
# Load Data
# =============================================================================
run_id = 513
DATA_PATH = f"/checkpoint/maui_sft/winnieyangwn/amaia_dumps/{run_id}/trajectories/{run_id}_metadata.jsonl"
log_dir  = "/checkpoint/maui_sft/winnieyangwn/rlm_dumps"
# Qwen3 model served via vLLM 
model_path = "Qwen/Qwen3-Coder-480B-A35B-Instruct" 
model_name = "qwen3" # Hugging Face model ID
# VLLM_HOST = os.getenv("VLLM_HOST", "localhost")  # Set via env var or default
VLLM_HOST = "h200-137-107-074" #  
VLLM_PORT = "32868"
VLLM_BASE_URL = f"http://{VLLM_HOST}:{VLLM_PORT}/v1"


def load_rollout_data(path: str, max_rows: int | None = None) -> list[dict]:
    """
    Load MLE Bench rollout data from JSONL file.
    
    Args:
        path: Path to the JSONL file
        max_rows: Optional limit on number of rows to load (for testing)
    
    Returns:
        List of rollout dictionaries
    """
    data = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if max_rows is not None and i >= max_rows:
                break
            data.append(json.loads(line))
    return data


def main():
    # Load the data (use max_rows for testing with smaller subset)
    print(f"Loading data from {DATA_PATH}...")
    context_data = load_rollout_data(DATA_PATH)  # Start with 100 rows for testing
    print(f"Loaded {len(context_data)} rollouts")

    # Build schema description with actual count
    data_schema = build_data_schema(len(context_data))

    # Set up logger
    logger = RLMLogger(log_dir=log_dir, file_name=f"{model_name}_mle_error_{run_id}")

    # Create the RLM Instance
    rlm = RLM(
        backend="vllm",
        backend_kwargs={
            "model_name": model_path,
            "base_url": VLLM_BASE_URL,
            "api_key": "not-needed",  # vLLM doesn't require an API key
        },
        environment="local",
        environment_kwargs={},
        max_depth=1,
        max_iterations=10,
        logger=logger,
        verbose=True,
    )

    # Define your question
    question = "What percentage of rollouts produced a valid submission?"

    # Build the root_prompt with data schema + question
    root_prompt = f"{data_schema}\nQUESTION: {question}"

    # Run RLM completion
    print(f"\nRunning RLM with question: {question}\n")
    result = rlm.completion(
        prompt=context_data,      # The data becomes the `context` variable in REPL
        root_prompt=root_prompt   # Schema + question shown at each iteration
    )



if __name__ == "__main__":
    main()
