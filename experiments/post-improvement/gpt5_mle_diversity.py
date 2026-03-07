"""
Diversity analysis of MLE Bench solution summaries with RLM.

This script demonstrates how to:
1. Load pre-generated solution summaries from a JSONL file
2. Use a diversity analysis prompt to analyze solution variety
3. Query the RLM to perform diversity analysis

Uses setup_code to load summarization data directly into REPL.
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
    "model_name": "gpt-5",
    "job_name": "post-improvement_diversity",
    "task_name": "",
    "summarization_file": "/checkpoint/maui_sft/winnieyangwn/rlm_dumps/post-improvement/gpt-5_post-improvement_summarization_520-rlm-comparison-argparse-xray_vinbigdata-chest-xray-abnormalities-detection_2026-02-10_21-02-49_42b6f38d_extracted.jsonl",
    "account": "agentic-models",
    "log_dir": "/checkpoint/maui_sft/winnieyangwn/rlm_dumps/post-improvement/",
    "run_id": "",
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze diversity of MLE Bench solution summaries with RLM")
    parser.add_argument("--account", type=str, default=DEFAULT_CONFIG["account"], help="Account name for data path")
    parser.add_argument("--model", type=str, default=DEFAULT_CONFIG["model_name"], help="Model name to use")
    parser.add_argument("--job_name", type=str, default=DEFAULT_CONFIG["job_name"], help="Job name for logging")
    parser.add_argument("--summarization_file", type=str, default=DEFAULT_CONFIG["summarization_file"], help="Path to summarization JSONL file")
    parser.add_argument("--log_dir", type=str, default=DEFAULT_CONFIG["log_dir"], help="Directory for log output")
    parser.add_argument("--max_depth", type=int, default=2, help="Max recursion depth for RLM")
    parser.add_argument("--run_id", type=str, default=DEFAULT_CONFIG["run_id"], help="Run ID for logging")
    parser.add_argument("--max_iterations", type=int, default=10, help="Max iterations for RLM")
    parser.add_argument("--verbose", action="store_true", default=True, help="Enable verbose output")
    parser.add_argument("--task_name", type=str, default=DEFAULT_CONFIG["task_name"], help="Task name for logging")
    return parser.parse_args()


def get_row_count(path: str) -> int:
    """Get number of rows in JSONL file."""
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


# =============================================================================
# Data Schema Description (for root_prompt)
# =============================================================================
def build_data_schema(num_summaries: int) -> str:
    return f"""
================================================================================
AVAILABLE VARIABLES (top-level, use directly - do NOT reassign these!)
================================================================================
The following variables are pre-loaded in the REPL namespace. Use them directly:
  - `summaries_df` (pandas.DataFrame) - Solution summaries from previous analysis
  - `pd` (module) - pandas is already imported

⚠️ WARNING: Do NOT call globals() or locals() - they are disabled.
⚠️ WARNING: Do NOT reassign these variables (e.g., `summaries_df = ...`).
   Just use them directly: `summaries_df.head()`, etc.

================================================================================
LLM HELPER FUNCTIONS (for sub-analysis)
================================================================================
  - `llm_query(prompt)` - Call the LLM with a single prompt (SLOW - sequential)
  - `llm_query_batched(prompts)` - Call the LLM with a list of prompts in PARALLEL (FAST)

================================================================================
SOLUTION SUMMARIES: `summaries_df`
================================================================================
A pandas DataFrame with {num_summaries} solution summaries from a previous summarization run.
Each row represents one solution's structured summary.

DATAFRAME COLUMNS:
├── solution_id: str              # Unique identifier for the solution
├── score_percentile: float       # Score 0-1 (higher = better, 1 = top)
├── data_preprocessing: str       # Data pipeline description
├── feature_engineering: str      # Feature engineering description
├── model_selection: str          # Model architecture and configuration
├── training_methodology: str     # Training approach and hyperparameters
├── evaluation_and_submission: str # Prediction and submission method
└── notable_implementation_details: str # Unique aspects of the implementation

ACCESS EXAMPLES:
  summaries_df["solution_id"].iloc[0]                 # First solution's ID
  summaries_df["model_selection"].iloc[0]             # First solution's model info
  summaries_df["data_preprocessing"].tolist()         # All preprocessing descriptions
  len(summaries_df)                                   # Number of solutions
"""


def build_question() -> str:
    """Build the diversity analysis question."""
    
    return """## Task

Analyze the diversity of code solutions using the pre-loaded solution summaries in `summaries_df`.

**How to access the data:**
- Each row in `summaries_df` contains a structured summary of one solution
- Key columns: `solution_id`, `score_percentile`, `data_preprocessing`, `feature_engineering`, 
  `model_selection`, `training_methodology`, `evaluation_and_submission`, `notable_implementation_details`

---

## Analysis Steps

1. **Review all solutions**: Read through all solution summaries in the DataFrame
2. **Analyze diversity**: Assess solution variety across multiple axes

### Diversity Axes to Analyze:
1. **Model Architecture Diversity**: What model families/architectures are used?
2. **Backbone Diversity**: What backbones/pretrained weights are used?
3. **Preprocessing Pipeline Diversity**: What are the distinct preprocessing strategies?
4. **Augmentation Diversity**: What augmentation strategies are used?
5. **Training Recipe Diversity**: What optimizer/scheduler/loss combinations?
6. **Post-processing Diversity**: What post-processing strategies are used?
7. **Ensemble Strategy Diversity**: Are ensembles used? What types?
8. **Validation Strategy Diversity**: What validation schemes are used?

---

## Output Format: Markdown Report

Write a comprehensive markdown report analyzing the diversity of solutions. Structure your report as follows:

```markdown
# Solution Diversity Analysis

## Overview
- Total number of solutions analyzed: X
- Summary of overall diversity

## Per-Axis Analysis

### 1. Model Architecture Diversity
- Assessment of diversity (high/medium/low)
- For each distinct architecture/family found:
  - **<Architecture Name>**: <count> solutions (X% of total) - IDs: [list of solution_ids]
  
Example format:
- **Faster R-CNN**: 45 solutions (82%) - IDs: [5, 6, 7, 8, ...]
- **YOLO**: 8 solutions (15%) - IDs: [1, 2, 3, 4, 10, 11, 12, 13]
- **RetinaNet**: 2 solutions (4%) - IDs: [9, 14]

### 2. Backbone Diversity
- Assessment of diversity (high/medium/low)
- For each distinct backbone/pretrained weights:
  - **<Backbone Name>**: <count> solutions (X% of total) - IDs: [list of solution_ids]

### 3. Preprocessing Pipeline Diversity
- Assessment of diversity (high/medium/low)
- For each distinct preprocessing technique:
  - **<Technique Name>**: <count> solutions (X% of total) - IDs: [list of solution_ids]

(Continue for all 8 axes with the same format: feature name, count, percentage, and solution IDs)

## Key Findings
- Most diverse axis: ...
- Most homogeneous axis: ...
- Notable patterns or clusters

## Redundant Solutions
- Identify pairs/groups of solutions that are near-duplicates across most axes
- Brief justification for each

## Gaps in Solution Space
- Notable approaches that are ABSENT but would be reasonable alternatives
- Suggestions for improving diversity
```

**IMPORTANT**: For each axis, report BOTH frequency counts AND solution IDs. Format: `**<Feature>**: <count> solutions (X%) - IDs: [id1, id2, ...]`

---

## IMPORTANT: Returning Your Final Answer

When you have completed your analysis:

1. **Store the markdown report (as a string) in a variable named exactly `final_answer`**
2. **Before returning, verify the variable exists** by printing: `print("final_answer" in dir())`
3. **Return using exactly**: `FINAL_VAR(final_answer)`

⚠️ Do NOT use a different variable name.
⚠️ Do NOT call FINAL_VAR with a variable that doesn't exist.

Example pattern:
```python
# Build your final answer as a markdown string
final_answer = \"\"\"
# Solution Diversity Analysis

## Overview
- Total number of solutions analyzed: 10
- Overall diversity assessment: ...

## Per-Axis Analysis
...
\"\"\"

# Verify it exists before returning
print("Variable 'final_answer' exists:", "final_answer" in dir())
```

Then in your next response, use: FINAL_VAR(final_answer)"""


def main() -> None:
    args = parse_args()
    
    # Validate summarization file exists
    summarization_path = validate_path(args.summarization_file, "Summarization file")

    # Get row count without loading data (fast)
    print(f"Counting rows in {summarization_path}...")
    num_summaries = get_row_count(str(summarization_path))
    print(f"Found {num_summaries} solution summaries")

    # Build schema description
    data_schema = build_data_schema(num_summaries)

    # Set up logger
    log_file_name = f"{args.model}_{args.job_name}_{args.run_id}_{args.task_name}"
    logger = RLMLogger(
        log_dir=args.log_dir,
        file_name=log_file_name
    )

    # Setup code: load summarization data directly into REPL
    setup_code = f"""
import pandas as pd

# Load solution summaries as DataFrame
summaries_df = pd.read_json('{summarization_path}', lines=True)
print(f"Loaded {{len(summaries_df)}} solution summaries")
print(f"Columns: {{summaries_df.columns.tolist()}}")
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
    question = build_question()
    root_prompt = f"{data_schema}\n\nQUESTION:\n{question}"

    # Run RLM completion
    print(f"\nRunning RLM diversity analysis (max_depth={args.max_depth}, max_iterations={args.max_iterations})...")
    print("(GPT-5 API calls may take 1-5+ minutes per iteration - please wait...)\n")
    result = rlm.completion(
        prompt="",
        root_prompt=root_prompt
    )



if __name__ == "__main__":
    main()
