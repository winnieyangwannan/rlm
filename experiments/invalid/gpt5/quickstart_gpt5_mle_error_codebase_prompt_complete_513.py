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
    
    Expected tags: <error_summary>, <root_cause>, <complete_revised_prompt>
    
    Returns:
        Dict mapping tag names to their content (stripped of whitespace)
    """
    sections = {}
    for tag in ["error_summary", "root_cause", "complete_revised_prompt"]:
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
CONFIG_YAML_PATH = "/home/winnieyangwn/amaia-collab/apps/sea/configs/winnieyang/eval/baseline/gpt5/513.yaml"  # Path to the YAML config file used to run the evaluation
PROMPT_PATH = "/home/winnieyangwn/amaia-collab/apps/sea/envs/envs/mle_bench/prompts/gpt5-517-rlm-complete.py"  # Path to the prompt file used in the evaluation
model_name = "gpt-5"  # Example model name
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

    # Generate timestamp for both logger and prompt output (ensures they match)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_name = f"{model_name}_{job_name}_{run_id}_{timestamp}"
    rlm_prompt_path = f"{log_dir}/{log_file_name}_prompt.py"

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
        max_depth=2,
        max_iterations=100,  # Reduced from 30 - simple analytical task
        logger=logger,
        verbose=True,
    )

    # Define your question
    # question = "What percentage of rollouts produced a valid submission?"
    # question = "Among all invalid submissions, what are the top 5 most common evaluation error messages?"
    question = """Your tasks are:

1. **Analyze Invalid Submissions**: Examine `rollout_df` and identify the top 5 most common evaluation error messages among all invalid submissions (where `valid_submission == False`). Show the error message and count for each.

2. **Codebase Investigation**: Search the `codebase` to identify which code components are responsible for this error. Is it a bug/mismatch/issue in the evaluation code, or is the agent producing invalid outputs?

3. **Diagnose Root Cause**: Focus on the most frequent errors. Analyze the relevant rollout transcripts (the `rollout` column) to understand what the agent was doing when this error occurred. What is the likely cause of this error?

4. **Prompt Revision**: This is the most important task.
   a) First, run `print(prompt)` to view the current prompt content in full
   b) Based on your analysis of the errors and root causes, revise the ENTIRE prompt to prevent these errors
   c) You may modify any part of the prompt: instructions, examples, formatting, constraints, etc.
   d) Make sure the revised prompt is complete and ready to use (not just a diff or partial changes)
   e) Preserve the overall structure and intent of the original prompt while improving it
   
   ⚠️ IMPORTANT FORMAT STRING RULE:
   - The prompt uses Python .format() for variable substitution
   - KEEP existing {variable} placeholders exactly as they are (single braces)
   - If you introduce ANY NEW variable placeholders that were NOT in the original prompt, 
     use DOUBLE curly braces {{new_variable}} so they escape to single braces after .format()
   - Example: Original has {task_description} → keep as {task_description}
   - Example: You add a new placeholder → write it as {{my_new_var}}

OUTPUT FORMAT:
Your final answer MUST be structured with these XML tags:


<top_5_error_summary>
Brief description of the top 5 errors (1-2 sentences each)
</top_5_error_summary>


<codebase_investigation>
Identify which codebase files/components are responsible for the error. Include:
- Relevant file paths and code snippets from the `codebase` dict
- Whether the issue is in evaluation code, agent output handling, or elsewhere
- Conclusion: Is this a bug in the evaluation infrastructure or invalid agent output?
</codebase_investigation>


<top_1_error_summary>
Brief description of the most frequent error (1-2 sentences). Show the error message and count for each.
</top_1_error_summary>


<top_1_error_root_cause>
Explanation of why the top 1 error occurs (2-3 sentences)
</top_1_error_root_cause>

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
    
    
    if parsed.get("top_1_error_root_cause"):
        print("\n=== ROOT CAUSE ===")
        print(parsed["top_1_error_root_cause"])
    
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
        print(result.response[:2000] + "..." if len(result.response) > 2000 else result.response)


if __name__ == "__main__":
    main()
