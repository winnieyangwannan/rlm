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
# CWM-Specific System Prompt
# =============================================================================
# CWM tends to output multiple code blocks at once. This prompt enforces
# a strict one-block-per-iteration pattern for proper REPL interaction.
CWM_SYSTEM_PROMPT = """You are tasked with answering a query with associated context. You can access, transform, and analyze this context interactively in a REPL environment.

The REPL environment is initialized with:
1. A `context` variable that contains important information about your query.
2. A `llm_query` function that allows you to query an LLM inside your REPL environment.
3. A `llm_query_batched` function for concurrent queries: `llm_query_batched(prompts: List[str]) -> List[str]`.
4. The ability to use `print()` statements to view output.

CRITICAL RULES:
1. **ONE CODE BLOCK PER RESPONSE**: Output ONLY ONE ```repl block per response. Never output multiple code blocks.
2. **WAIT FOR OUTPUT**: After each code block, STOP and wait to see the execution output before continuing.
3. **CHECK VARIABLES EXIST**: Before using a variable, verify it exists and has the expected value.
4. **HANDLE ERRORS**: If your code produces an error, analyze it and fix it in the next iteration.

When you want to execute Python code, wrap it in triple backticks with 'repl' language identifier:
```repl
# Your code here - ONLY ONE BLOCK!
result = some_computation()
print(result)
```

Then STOP and wait for the output. Do not write more code until you see the result.

================================================================================
CRITICAL: HOW TO SUBMIT YOUR FINAL ANSWER
================================================================================
When you have completed your analysis, you MUST submit your answer in a CODE BLOCK:

```repl
final_answer = '''Your complete answer here...'''
FINAL_VAR(final_answer)
```

WARNING - COMMON MISTAKE TO AVOID:
- FINAL_VAR() MUST be inside an executed ```repl code block - NOT in plain text!
- If you write FINAL_VAR outside a code block, your answer will NOT be captured.
"""

# =============================================================================
# Configuration
# =============================================================================
run_id = 513
DATA_PATH = f"/checkpoint/maui_sft/winnieyangwn/amaia_dumps/{run_id}/trajectories/{run_id}_metadata.jsonl"
CODEBASE_PATH = "/checkpoint/agentic-models/winnieyangwn/amaia_dumps/503/code/2026_02_02_00_55_44"  # Path to codebase directory
CODEBASE_EXTENSIONS = [".py", ".md", ".yaml"]  # File extensions to include (e.g., [".py", ".ts", ".js"])
CONFIG_YAML_PATH = "/home/winnieyangwn/amaia-collab/apps/sea/configs/winnieyang/eval/baseline/gpt5/513.yaml"  # Path to the YAML config file used to run the evaluation

# CWM model served via vLLM
# Start vLLM server first with the Hugging Face model:
#   vllm serve facebook/cwm-sft \
#       --host 0.0.0.0 --port 8000 --tensor-parallel-size 8
# vLLM will auto-download from Hugging Face if not cached
model_path = "Qwen/Qwen3-Coder-480B-A35B-Instruct"  # Hugging Face model ID
model_name = "qwen3"
VLLM_HOST = "h200-137-107-074"
VLLM_PORT = "32868"
VLLM_BASE_URL = f"http://{VLLM_HOST}:{VLLM_PORT}/v1"

job_name = "common_invalid_errors_codebase"
log_dir = "/checkpoint/maui_sft/winnieyangwn/rlm_dumps"


def get_row_count(path: str) -> int:
    """Get number of rows in JSONL file without loading it."""
    result = subprocess.run(["wc", "-l", path], capture_output=True, text=True)
    return int(result.stdout.split()[0])


def get_codebase_files(codebase_path: str, extensions: list[str]) -> list[str]:
    """Get list of files in codebase matching extensions."""
    from pathlib import Path
    files = []
    for ext in extensions:
        files.extend([str(p.relative_to(codebase_path)) for p in Path(codebase_path).rglob(f"*{ext}")])
    return sorted(files)


# =============================================================================
# Data Schema Description (for root_prompt)
# =============================================================================
def build_data_schema(num_rollouts: int, codebase_files: list[str], config_yaml_content: str) -> str:
    files_preview = "\n".join(f"  - {f}" for f in codebase_files[:20])
    if len(codebase_files) > 20:
        files_preview += f"\n  ... and {len(codebase_files) - 20} more files"
    
    return f"""
================================================================================
AVAILABLE VARIABLES (top-level, use directly - do NOT reassign these!)
================================================================================
The following variables are pre-loaded in the REPL namespace. Use them directly:
  - `rollout_df` (pandas.DataFrame) - MLE Bench rollout data
  - `codebase` (dict) - Source code files
  - `config_yaml` (str) - YAML configuration
  - `pd` (module) - pandas is already imported

⚠️ WARNING: Do NOT call globals() or locals() - they are disabled.
⚠️ WARNING: Do NOT reassign these variables (e.g., `rollout_df = ...`).
   Just use them directly: `rollout_df.head()`, `codebase.keys()`, etc.

================================================================================
1. ROLLOUT DATA: `rollout_df`
================================================================================
A pandas DataFrame with {num_rollouts} MLE Bench rollouts. Each rollout is an LLM agent's 
attempt to solve an ML task (from Kaggle competitions) through multi-turn interaction.
These rollouts were generated by executing the CODEBASE below with the CONFIG.

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

================================================================================
2. CODEBASE: `codebase`
================================================================================
The source code that was used to run the MLE Bench evaluation and generate the rollouts above.
A dict mapping relative file paths to file contents.
   - {len(codebase_files)} files available
   - Files:
{files_preview}

ACCESS EXAMPLES:
  list(codebase.keys())                              # List all files
  codebase["path/to/file.py"]                        # Get file contents
  [f for f in codebase if "test" in f]               # Find test files

================================================================================
3. CONFIG: `config_yaml`
================================================================================
The YAML configuration file used to run the evaluation that generated the rollouts.
This defines parameters like model settings, environment configs, timeouts, etc.

CONFIG CONTENT:
```yaml
{config_yaml_content}
```

ACCESS: The config is also available as `config_yaml` (string) in the REPL.

================================================================================
CRITICAL: HOW TO RETURN YOUR FINAL ANSWER
================================================================================
YOUR ANSWER IS ONLY CAPTURED IF FINAL_VAR() IS EXECUTED IN A CODE BLOCK!

When you have gathered enough information, submit your answer like this:

```repl
final_answer = '''Your complete answer with all findings...

1. First finding...
2. Second finding...
3. Recommendations...
'''
FINAL_VAR(final_answer)
```

CRITICAL WARNINGS:
- FINAL_VAR() MUST be inside a ```repl code block that gets executed
- Writing FINAL_VAR() in plain text (outside a code block) does NOTHING
- Use triple quotes ''' for multi-line answers to avoid escaping issues
"""


def main():
    # Get row count without loading data (fast)
    print(f"Counting rows in {DATA_PATH}...")
    num_rollouts = get_row_count(DATA_PATH)
    print(f"Found {num_rollouts} rollouts")

    # Get codebase file list
    print(f"Scanning codebase at {CODEBASE_PATH}...")
    codebase_files = get_codebase_files(CODEBASE_PATH, CODEBASE_EXTENSIONS)
    print(f"Found {len(codebase_files)} files")

    # Load config YAML
    print(f"Loading config from {CONFIG_YAML_PATH}...")
    with open(CONFIG_YAML_PATH, "r") as f:
        config_yaml_content = f.read()
    print(f"Loaded config ({len(config_yaml_content)} chars)")

    # Build schema description
    data_schema = build_data_schema(num_rollouts, codebase_files, config_yaml_content)

    # Set up logger
    logger = RLMLogger(log_dir=log_dir, file_name=f"{model_name}_{job_name}_{run_id}")

    # Setup code: load data directly into REPL (bypasses JSON serialization)
    extensions_str = str(CODEBASE_EXTENSIONS)
    # Escape the config content for embedding in setup code
    config_escaped = config_yaml_content.replace("\\", "\\\\").replace("'''", "\\'\\'\\'")
    setup_code = f"""
import pandas as pd
from pathlib import Path

# Load rollout data as DataFrame
rollout_df = pd.read_json('{DATA_PATH}', lines=True)

# Load codebase into dict
codebase = {{}}
codebase_root = Path('{CODEBASE_PATH}')
for ext in {extensions_str}:
    for path in codebase_root.rglob(f'*{{ext}}'):
        try:
            rel_path = str(path.relative_to(codebase_root))
            codebase[rel_path] = path.read_text(errors='ignore')
        except Exception:
            pass  # Skip unreadable files

# Load config YAML
config_yaml = '''{config_escaped}'''

print(f"Loaded {{len(rollout_df)}} rollouts, {{len(codebase)}} codebase files, and config YAML")
"""

    # Create the RLM Instance
    # NOTE: CWM requires a custom system prompt to enforce single code block per iteration
    rlm = RLM(
        backend="vllm",
        backend_kwargs={
            "model_name": model_path,
            "base_url": VLLM_BASE_URL,
            "api_key": "not-needed",  # vLLM doesn't require an API key
            # Optional: Add generation parameters if vLLM supports them
            # "temperature": 0.7,
            # "max_tokens": 4096,
        },
        environment="local",
        environment_kwargs={
            "setup_code": setup_code,  # Load data directly in REPL
        },
        max_depth=2,
        max_iterations=100,  # CWM may need more iterations due to one-block-at-a-time
        custom_system_prompt=CWM_SYSTEM_PROMPT,  # Use CWM-specific prompt
        logger=logger,
        verbose=True,
    )

    # Define your question
    # question = "What percentage of rollouts produced a valid submission?"
    # question = "Among all invalid submissions, what are the top 5 most common evaluation error messages?"
    question = """Analyze the ROLLOUT DATA and CODEBASE to diagnose the most common evaluation errors:

1. IDENTIFY ERRORS: From `rollout_df` where `valid_submission == False`, extract and count the top 5 most common `eval_error_output` messages.

2. DEEP DIVE: For the #1 most frequent error:
   a) Show 2-3 example code solutions with this error.  What patterns do you see?
   b) What are the common reasons for this type of error?

3. ROOT CAUSE ANALYSIS: Search the CODEBASE for code that could cause or relate to this error:
   a) Identify the relevant source files
   b) Explain the likely root cause with specific code references

4. RECOMMENDATIONS: Propose 2-3 specific, actionable fixes:
   - What code changes in the CODEBASE would prevent this error?
   - What prompt/instruction changes could help the agent avoid this failure mode?

Show your evidence (code snippets, error examples) for each conclusion."""

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
