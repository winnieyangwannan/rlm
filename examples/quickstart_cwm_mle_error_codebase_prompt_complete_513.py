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
import re
import subprocess
from datetime import datetime

from dotenv import load_dotenv

from rlm import RLM
from rlm.logger import RLMLogger

load_dotenv()


# =============================================================================
# Output Parsing Utilities
# =============================================================================
def parse_agent_output(response: str) -> dict[str, str]:
    """
    Parse structured XML sections from agent response.
    
    Expected tags: <error_analysis>, <complete_revised_prompt>
    
    Returns:
        Dict mapping tag names to their content (stripped of whitespace)
    """
    sections = {}
    for tag in ["error_analysis", "complete_revised_prompt"]:
        pattern = rf"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            sections[tag] = match.group(1).strip()
    return sections


def extract_format_variables(text: str) -> set[str]:
    """
    Extract all {variable} format placeholders from text.
    
    Ignores already-escaped {{variable}} patterns.
    Returns set of variable names.
    """
    # Match {word} but not {{word}} (already escaped)
    # First, temporarily replace {{ and }} to avoid matching them
    temp_text = text.replace("{{", "\x00\x00").replace("}}", "\x01\x01")
    # Find all {variable} patterns (simple identifiers)
    matches = re.findall(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", temp_text)
    return set(matches)


def escape_new_format_variables(original_prompt: str, revised_prompt: str) -> str:
    """
    Escape any new {variable} placeholders in revised_prompt that weren't in original_prompt.
    
    New variables are replaced with {{variable}} to prevent KeyError when .format() is called.
    
    Args:
        original_prompt: The original prompt template
        revised_prompt: The revised prompt that may contain new variables
        
    Returns:
        The revised prompt with new variables escaped as {{variable}}
    """
    original_vars = extract_format_variables(original_prompt)
    revised_vars = extract_format_variables(revised_prompt)
    
    new_vars = revised_vars - original_vars
    
    if not new_vars:
        return revised_prompt
    
    print(f"\n[WARNING] Found {len(new_vars)} new format variable(s) not in original prompt: {new_vars}")
    print("[INFO] Escaping them with double braces {{...}} to prevent KeyError")
    
    result = revised_prompt
    for var in new_vars:
        # Replace {var} with {{var}}, but be careful not to replace already-escaped {{var}}
        # Use negative lookbehind and lookahead to avoid {{var}}
        pattern = r"(?<!\{)\{" + re.escape(var) + r"\}(?!\})"
        result = re.sub(pattern, "{{" + var + "}}", result)
    
    return result

# =============================================================================
# Configuration
# =============================================================================
run_id = 513
DATA_PATH = f"/checkpoint/maui_sft/winnieyangwn/amaia_dumps/{run_id}/trajectories/{run_id}_metadata.jsonl"
CODEBASE_PATH = "/checkpoint/agentic-models/winnieyangwn/amaia_dumps/503/code/2026_02_02_00_55_44"  # Path to codebase directory
CODEBASE_EXTENSIONS = [".py", ".md", ".yaml"]  # File extensions to include (e.g., [".py", ".ts", ".js"])
MAX_CODEBASE_CHARS = 50000  # Limit total codebase content to avoid exceeding context length
MAX_FILE_CHARS = 5000  # Limit individual file size
CONFIG_YAML_PATH = "/home/winnieyangwn/amaia-collab/apps/sea/configs/winnieyang/eval/baseline/gpt5/513.yaml"  # Path to the YAML config file used to run the evaluation
PROMPT_PATH = "/home/winnieyangwn/amaia-collab/apps/sea/envs/envs/mle_bench/prompts/gpt5-517-rlm-complete.py"  # Path to the prompt file used in the evaluation

# CWM model served via vLLM
# Start vLLM server first with the Hugging Face model:
#   vllm serve facebook/cwm-sft \
#       --host 0.0.0.0 --port 8000 --tensor-parallel-size 8
# vLLM will auto-download from Hugging Face if not cached
model_path = "facebook/cwm-sft"
model_name = "cwm"  # Hugging Face model ID
VLLM_BASE_URL = f"http://h200-061-030:9255/v1"

job_name = "common_invalid_errors_codebase_prompt_complete"
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
def build_data_schema(num_rollouts: int, codebase_files: list[str], config_yaml_content: str, prompt_content: str) -> str:
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
  - `prompt` (str) - The prompt used in the evaluation
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
4. PROMPT: `prompt`
================================================================================
The prompt template used to instruct the LLM agent during the evaluation.
This is the system/user prompt that guides the agent's behavior.

PROMPT CONTENT:
```
{prompt_content}
```

ACCESS: The prompt is also available as `prompt` (string) in the REPL.

================================================================================
IMPORTANT: HOW TO RETURN YOUR FINAL ANSWER
================================================================================
When returning your final answer, ALWAYS use FINAL_VAR instead of FINAL to avoid parsing issues:

1. Store your answer in a variable first:
   final_answer = "Your complete answer here..."

2. Then return it using FINAL_VAR:
   FINAL_VAR(final_answer)

DO NOT use FINAL(...) directly with content containing parentheses like "1)" or "2)" 
as this will truncate your answer.
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

    # Load prompt
    print(f"Loading prompt from {PROMPT_PATH}...")
    with open(PROMPT_PATH, "r") as f:
        prompt_content = f.read()
    print(f"Loaded prompt ({len(prompt_content)} chars)")

    # Build schema description
    data_schema = build_data_schema(num_rollouts, codebase_files, config_yaml_content, prompt_content)

    # Generate log file name (RLMLogger adds timestamp and run_id automatically)
    log_file_name = f"{model_name}_{job_name}_{run_id}"
    # Generate timestamp for prompt output file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    rlm_prompt_path = f"{log_dir}/{log_file_name}_{timestamp}_prompt.py"

    # Set up logger
    logger = RLMLogger(log_dir=log_dir, file_name=log_file_name)

    # Setup code: load data directly into REPL (bypasses JSON serialization)
    extensions_str = str(CODEBASE_EXTENSIONS)
    # Escape the config content for embedding in setup code
    config_escaped = config_yaml_content.replace("\\", "\\\\").replace("'''", "\\'\\'\\'")
    # Escape the prompt content for embedding in setup code
    prompt_escaped = prompt_content.replace("\\", "\\\\").replace("'''", "\\'\\'\\'")
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

# Load prompt
prompt = '''{prompt_escaped}'''

print(f"Loaded {{len(rollout_df)}} rollouts, {{len(codebase)}} codebase files, config YAML, and prompt")
"""

    # Create the RLM Instance
    rlm = RLM(
        backend="vllm",
        backend_kwargs={
            "model_name": model_path,
            "base_url": VLLM_BASE_URL,
            "api_key": "not-needed",  # vLLM doesn't require an API key
            # Enable thinking for CWM model via vLLM chat_template_kwargs
            "extra_body": {
                "chat_template_kwargs": {
                    "enable_thinking": True,
                    "preserve_previous_think": True,
                }
            },
        },
        environment="local",
        environment_kwargs={
            "setup_code": setup_code,  # Load data directly in REPL
        },
        max_depth=2,
        max_iterations=15,  # Reduced to avoid exceeding context length (~10K tokens/iteration)
        logger=logger,
        verbose=True,
    )

    # Define your question
    # question = "What percentage of rollouts produced a valid submission?"
    # question = "Among all invalid submissions, what are the top 5 most common evaluation error messages?"
    question = """
================================================================================
PACING INSTRUCTIONS (READ FIRST!)
================================================================================
You have UP TO 15 ITERATIONS to complete this task. Take your time!

DO NOT try to do everything in one iteration. Instead:
- Iteration 1: Identify the top 5 errors and their counts
- Iteration 2: Analyze Error #1 (codebase search + rollout analysis)
- Iteration 3: Analyze Error #2
- Iteration 4: Analyze Error #3
- Iteration 5: Analyze Error #4
- Iteration 6: Analyze Error #5
- Iteration 7: Review the prompt with print(prompt)
- Iteration 8-10: Draft and refine the revised prompt
- Final iteration: Output your final answer with FINAL_VAR()

Each iteration, focus on ONE thing. Print your findings, then stop and wait for the next iteration.

================================================================================
YOUR TASKS
================================================================================

**Task 1: Identify Top 5 Errors** (do this FIRST, in iteration 1)
Examine `rollout_df` where `valid_submission == False`. 
List the top 5 most common `eval_error_output` messages with their counts.
```repl
# Run this first, then STOP and wait for next iteration
error_counts = rollout_df[rollout_df['valid_submission'] == False]['eval_error_output'].value_counts().head(5)
print(error_counts)
```

**Task 2: Deep Analysis of Each Error** (one error per iteration!)
For EACH of the top 5 errors (ONE AT A TIME, one per iteration):
   a) **Codebase Search**: Search the `codebase` dict for the error string
   b) **Rollout Analysis**: Examine 2-3 rollouts where this error occurred
   c) **Root Cause**: Agent behavior, eval bug, or environment issue?
   d) **Prompt Fix**: What specific change to the prompt would help?

**Task 3: Prompt Revision** (after analyzing all 5 errors)
   a) Run `print(prompt)` to view the current prompt
   b) Revise the ENTIRE prompt to prevent these errors
   c) Make it complete and ready to use
   
   ⚠️ FORMAT STRING RULE:
   - KEEP existing {variable} placeholders as-is (single braces)
   - NEW placeholders must use DOUBLE braces {{new_var}}

================================================================================
OUTPUT FORMAT (only in your FINAL answer)
================================================================================
Your final answer MUST be structured with these XML tags:

<error_analysis>
## Error 1: [exact error message] (count: N)
**Codebase Location**: [file path(s) and relevant code snippet]
**Rollout Pattern**: [what the agent was doing in the last 2-3 turns before error]
**Root Cause**: [agent behavior / eval bug / environment issue - explain why]
**Prompt Fix Needed**: [specific instruction to add or modify]

## Error 2: [exact error message] (count: N)
**Codebase Location**: [file path(s) and relevant code snippet]
**Rollout Pattern**: [what the agent was doing in the last 2-3 turns before error]
**Root Cause**: [agent behavior / eval bug / environment issue - explain why]
**Prompt Fix Needed**: [specific instruction to add or modify]

## Error 3: [exact error message] (count: N)
**Codebase Location**: [file path(s) and relevant code snippet]
**Rollout Pattern**: [what the agent was doing in the last 2-3 turns before error]
**Root Cause**: [agent behavior / eval bug / environment issue - explain why]
**Prompt Fix Needed**: [specific instruction to add or modify]

## Error 4: [exact error message] (count: N)
**Codebase Location**: [file path(s) and relevant code snippet]
**Rollout Pattern**: [what the agent was doing in the last 2-3 turns before error]
**Root Cause**: [agent behavior / eval bug / environment issue - explain why]
**Prompt Fix Needed**: [specific instruction to add or modify]

## Error 5: [exact error message] (count: N)
**Codebase Location**: [file path(s) and relevant code snippet]
**Rollout Pattern**: [what the agent was doing in the last 2-3 turns before error]
**Root Cause**: [agent behavior / eval bug / environment issue - explain why]
**Prompt Fix Needed**: [specific instruction to add or modify]
</error_analysis>

<complete_revised_prompt>
The COMPLETE revised prompt that should replace the original prompt.
This must be the full prompt text, not just the changes.
Include all sections from the original prompt with your improvements.
The prompt should be ready to use directly without any additional editing.
</complete_revised_prompt>
"""

    # Build the root_prompt with data schema + question
    root_prompt = f"{data_schema}\nQUESTION: {question}"

    # Run RLM completion
    print(f"\nRunning RLM with question: {question}\n")
    result = rlm.completion(
        prompt="",                # Empty - data loaded via setup_code
        root_prompt=root_prompt   # Schema + question shown at each iteration
    )

    # ==========================================================================
    # Parse structured output
    # ==========================================================================
    print("\n" + "=" * 80)
    print("PARSED OUTPUT:")
    print("=" * 80)
    
    parsed = parse_agent_output(result.response)
    
    # if parsed.get("error_analysis"):
    #     print("\n=== ERROR ANALYSIS (ALL 5 ERRORS) ===")
    #     print(parsed["error_analysis"][:3000] + "..." if len(parsed["error_analysis"]) > 3000 else parsed["error_analysis"])
    
    if parsed.get("complete_revised_prompt"):
        print("\n=== COMPLETE REVISED PROMPT ===")
        print(parsed["complete_revised_prompt"][:1000] + "..." if len(parsed["complete_revised_prompt"]) > 1000 else parsed["complete_revised_prompt"])
        
        # Escape any new format variables not in the original prompt
        revised_prompt_safe = escape_new_format_variables(prompt_content, parsed["complete_revised_prompt"])
        
        # Save the complete revised prompt as a .py file
        with open(rlm_prompt_path, "w") as f:
            f.write(revised_prompt_safe)
        print(f"\n[Saved complete revised prompt to: {rlm_prompt_path}]")
    else:
        print("\n[WARNING] No <complete_revised_prompt> section found in response.")
        print("Raw response preview:")
        # print(result.response[:2000] + "..." if len(result.response) > 2000 else result.response)


if __name__ == "__main__":
    main()
