"""
Comparison pipeline for analyzing MLE Bench rollout data with RLM.

This script performs comparison analysis given existing summarization files:
- Load summarization files → compare/aggregate solutions → final insights

Requires: Completed summarization files (from gpt5_mle_summarization.py or similar)
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from dotenv import load_dotenv

from rlm import RLM
from rlm.logger import RLMLogger

if TYPE_CHECKING:
    from rlm.core.types import RLMChatCompletion

load_dotenv()


# =============================================================================
# Helper functions to be injected into REPL via setup_code
# =============================================================================


def assemble_summary(row: dict) -> str:
    """Assemble a formatted markdown summary from a SUMMARIES row.
    
    Args:
        row: A single row from SUMMARIES (e.g., SUMMARIES.iloc[0].to_dict() or a dict)
    
    Returns:
        Formatted markdown string with all fields
    """
    parts = []
    
    # Header with rollout info
    rollout_id = row.get("rollout_id", "N/A")
    percentile = row.get("score_percentile", "N/A")
    medal = row.get("medal_earned", None)
    medal_str = f" | Medal: {medal}" if medal else ""
    if isinstance(percentile, (int, float)):
        parts.append(f"# Rollout {rollout_id} (Percentile: {percentile:.2f}{medal_str})")
    else:
        parts.append(f"# Rollout {rollout_id} (Percentile: {percentile}{medal_str})")
    parts.append("")
    
    # Summary
    if row.get("summary"):
        parts.append("## Summary")
        parts.append(row["summary"])
        parts.append("")
    
    # Task Analysis
    if row.get("task_analysis"):
        parts.append("## Task Analysis")
        parts.append(row["task_analysis"])
        parts.append("")
    
    # Data Preprocessing
    if row.get("data_preprocessing"):
        parts.append("## Data Preprocessing")
        parts.append(row["data_preprocessing"])
        parts.append("")
    
    # Feature Engineering
    if row.get("feature_engineering"):
        parts.append("## Feature Engineering")
        parts.append(row["feature_engineering"])
        parts.append("")
    
    # Model Selection
    if row.get("model_selection"):
        parts.append("## Model Selection")
        parts.append(row["model_selection"])
        parts.append("")
    
    # Training Methodology
    if row.get("training_methodology"):
        parts.append("## Training Methodology")
        parts.append(row["training_methodology"])
        parts.append("")
    
    # Evaluation and Submission
    if row.get("evaluation_and_submission"):
        parts.append("## Evaluation and Submission")
        parts.append(row["evaluation_and_submission"])
        parts.append("")
    
    # Notable Implementation Details
    if row.get("notable_implementation_details"):
        parts.append("## Notable Implementation Details")
        parts.append(row["notable_implementation_details"])
        parts.append("")
    
    return "\n".join(parts)


# =============================================================================
# Configuration (defaults - can be overridden via CLI)
# =============================================================================
DEFAULT_CONFIG = {
    "run_id": 514,
    "model_name": "gpt-5",
    "job_name": "comparison",
    "task_name": "tweet-sentiment-extraction",  # Used in log file naming; does not affect which summarization files are loaded (those are determined by --summarization_dir and the presence of .md files)
    "log_dir": "/checkpoint/maui_sft/winnieyangwn/rlm_dumps/comparison/mle-30/v6",
    "summarization_dir": "/checkpoint/maui_sft/winnieyangwn/rlm_dumps/summarization/mle-30/v6",
}



def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare MLE Bench solutions given summarization files")
    parser.add_argument("--summarization_dir", type=str, default=DEFAULT_CONFIG["summarization_dir"], help="Directory containing summarization .md files")
    parser.add_argument("--run_id", type=int, default=DEFAULT_CONFIG["run_id"], help="Run ID to analyze")
    parser.add_argument("--model", type=str, default=DEFAULT_CONFIG["model_name"], help="Model name to use")
    parser.add_argument("--job_name", type=str, default=DEFAULT_CONFIG["job_name"], help="Job name for logging")
    parser.add_argument("--task_name", type=str, default="tweet-sentiment-extraction", help="Task name to analyze")
    parser.add_argument("--log_dir", type=str, default=DEFAULT_CONFIG["log_dir"], help="Directory to save output logs")
    parser.add_argument("--max_depth", type=int, default=2, help="Max recursion depth for RLM")
    parser.add_argument("--max_iterations", type=int, default=20, help="Max iterations for RLM")
    parser.add_argument("--verbose", action="store_true", default=True, help="Enable verbose output")
    return parser.parse_args()



# Medal Solution Deep Dive prompt - used in both template and SETUP_CODE
MEDAL_DEEP_DIVE_PROMPT = """
Note: If fewer than 2 medal solutions exist, treat 🔵 Very High solutions as the reference tier throughout this section.

For each medal solution, analyze:
- **Key decisions**: The 2-3 choices that most explain its medal-level performance — be specific about *why* they mattered given this task's structure
"""

# Critical Failure Analysis prompt - used in template and SETUP_CODE
CRITICAL_FAILURE_PROMPT = """Identify patterns strongly associated with 🔴 Low tier:
- What mistakes or omissions most reliably produce low scores?
- Are there "trap" approaches that seem reasonable but fail on this specific task?
- What are the important things to get right to avoid catastrophic failure?"""

# Dimension-specific analysis template (used by individual dimension steps)
DIMENSION_ANALYSIS_TEMPLATE = """Structure your analysis as:

- **Key tier differences**: What distinguishes high-performing tiers from low-performing ones. Include frequency counts (e.g. "4/5 medal solutions used X, vs 1/8 low-tier solutions").
- **Critical transitions**:
  - *Low → Medium*: Minimum changes needed to escape the low tier
  - *Medium → High*: What unlocks above-average performance
  - *High → Medal*: What separates near-medal from medal (combine Very High and Medal if patterns are similar)
- **Why this matters for this task**: Connect the pattern to a specific property of the dataset, metric, or task structure. If the mechanism is unclear, note it as correlation only.
- **Generalizability**: Is this pattern likely specific to this task, or likely to transfer across similar competitions?
"""

# Individual dimension prompts - Steps 4a-4f
DATA_PREPROCESSING_PROMPT = f"""## Task: Contrastive Analysis - Data Preprocessing

Analyze how DATA PREPROCESSING differs across tiers.

Focus on: cleaning, transformations, pipeline structure, augmentation strategies.

{DIMENSION_ANALYSIS_TEMPLATE}

Use REPL to examine the `data_preprocessing` column across tiers.

When done, store your markdown analysis in a variable and call `FINAL_VAR(variable_name)`.
"""

FEATURE_ENGINEERING_PROMPT = f"""## Task: Contrastive Analysis - Feature Engineering

Analyze how FEATURE ENGINEERING differs across tiers.

Focus on: new features created, feature selection/reduction techniques.

{DIMENSION_ANALYSIS_TEMPLATE}

Use REPL to examine the `feature_engineering` column across tiers.

When done, store your markdown analysis in a variable and call `FINAL_VAR(variable_name)`.
"""

MODEL_SELECTION_PROMPT = f"""## Task: Contrastive Analysis - Model Selection

Analyze how MODEL SELECTION differs across tiers.

Focus on: algorithms used, hyperparameters, ensemble methods, pretrained models.

{DIMENSION_ANALYSIS_TEMPLATE}

Use REPL to examine the `model_selection` column across tiers.

When done, store your markdown analysis in a variable and call `FINAL_VAR(variable_name)`.
"""

TRAINING_METHODOLOGY_PROMPT = f"""## Task: Contrastive Analysis - Training Methodology

Analyze how TRAINING METHODOLOGY differs across tiers.

Focus on: objective alignment, validation strategy, training configuration, tuning approaches.

{DIMENSION_ANALYSIS_TEMPLATE}

Use REPL to examine the `training_methodology` column across tiers.

When done, store your markdown analysis in a variable and call `FINAL_VAR(variable_name)`.
"""

EVALUATION_SUBMISSION_PROMPT = f"""## Task: Contrastive Analysis - Evaluation & Submission

Analyze how EVALUATION AND SUBMISSION differs across tiers.

Focus on: prediction methods, post-processing techniques, submission strategies.

{DIMENSION_ANALYSIS_TEMPLATE}

Use REPL to examine the `evaluation_and_submission` column across tiers.

When done, store your markdown analysis in a variable and call `FINAL_VAR(variable_name)`.
"""

IMPLEMENTATION_DETAILS_PROMPT = f"""## Task: Contrastive Analysis - Notable Implementation Details

Analyze how NOTABLE IMPLEMENTATION DETAILS differ across tiers.

Focus on: unique techniques, clever optimizations, domain-specific considerations.

{DIMENSION_ANALYSIS_TEMPLATE}

Use REPL to examine the `notable_implementation_details` column across tiers.

When done, store your markdown analysis in a variable and call `FINAL_VAR(variable_name)`.
"""

# Actionable Recommendations prompt - Step 5
RECOMMENDATIONS_PROMPT = """## Task: Actionable Recommendations

Based on the prior analysis, provide concrete, prioritized advice for three ambition levels:

- 🥇 **Target Medal**: What would a solution need to do differently from the typical High-tier solution?
- 🟢 **Target High+**: What are the highest-leverage improvements over a Medium solution?
- ✅ **Minimum viable**: What are the non-negotiable requirements to avoid the Low tier?

Be specific and reference patterns identified in the prior analysis.

When done, store your markdown analysis in a variable and call `FINAL_VAR(variable_name)`.
"""

# (CONTRASTIVE_REFLECTION_TEMPLATE removed - now using multi-step pipeline)

comparison_schema = f"""
================================================================================
AVAILABLE VARIABLES
================================================================================
  - `SUMMARIES`: DataFrame containing parsed rollout summaries (NOTE: use this instead of `context`, which is empty). Columns:
    - `rollout_id`: int - Rollout identifier
    - `score_percentile`: float - Score percentile (0-1)
    - `medal_earned`: str or None - Medal type (gold/silver/bronze) or None
    - `task_analysis`: str - Flattened string with goal, task_type, data_modality, evaluation_metric, core_challenges, difficulty_factors
    - `summary`: str - 2-3 sentence overview of the solution
    - `data_preprocessing`: str - Flattened string with cleaning, transformations, pipeline_structure, augmentation
    - `feature_engineering`: str - Flattened string with new_features, selection_or_reduction
    - `model_selection`: str - Flattened string with algorithms, hyperparameters, ensemble, pretrained_models
    - `training_methodology`: str - Flattened string with objective_alignment, validation_strategy, training_configuration, tuning
    - `evaluation_and_submission`: str - Flattened string with prediction_method, post_processing
    - `notable_implementation_details`: str or None - Notable techniques and considerations
    - `assembled`: str - **PRE-COMPUTED** full markdown summary combining all fields above. Use this for passing to llm_query.
  - `MEDAL_DEEP_DIVE_PROMPT`: str - Analysis prompt for medal solution deep dive (use in llm_query and llm_query_batched calls)
  - `CRITICAL_FAILURE_PROMPT`: str - Analysis prompt for identifying low-tier failure patterns (use in llm_query and llm_query_batched calls)
  - `assemble_summary(row)`: Function that takes a SUMMARIES row (dict) and returns a formatted markdown string with all fields. Usage: `assemble_summary(SUMMARIES.iloc[0].to_dict())`. NOTE: The `assembled` column is already pre-computed using this function.
  - `pd`: pandas is already imported

================================================================================
AVAILABLE FUNCTIONS
================================================================================
  - `llm_query(prompt)`: Query a sub-LLM with a single prompt string. Returns response string.
  - `llm_query_batched(prompts)`: Query sub-LLM with multiple prompts in parallel. Pass list of strings, returns list of responses. **Preferred for efficiency.**
  - `FINAL_VAR(variable_name)`: Return a REPL variable as your final answer. Call only when analysis is complete.
"""


def load_summarization_folder(summarization_dir: str, task_name: str) -> Path | None:
    """Find summarization folder for a given task.
    
    Args:
        summarization_dir: Base directory containing task subfolders with .md files
        task_name: Task name to find the folder for
        
    Returns:
        Path to the task folder containing .md files, or None if not found
    """
    # Task folder is at summarization_dir/task_name/
    task_folder = Path(summarization_dir) / task_name
    
    if not task_folder.exists():
        print(f"Task folder not found: {task_folder}")
        return None
    
    # Count .md files in the folder
    md_files = list(task_folder.glob("*.md"))
    
    if not md_files:
        print(f"No .md files found in task folder: {task_folder}")
        return None
    
    print(f"Found {len(md_files)} .md files in: {task_folder}")
    return task_folder


def assign_tier(row: dict) -> str:
    """Assign performance tier based on score_percentile and medal_earned.
    
    Tier criteria:
        🥇 Medal: Medal-winning solutions
        🔵 Very High: percentile >= 0.9, no medal
        🟢 High: 0.7 <= percentile < 0.9
        🟡 Medium: 0.4 <= percentile < 0.7
        🔴 Low: percentile < 0.4
    
    Args:
        row: A dict with 'score_percentile' and 'medal_earned' keys
        
    Returns:
        Tier label string (e.g., "🥇 Medal", "🔵 Very High", etc.)
    """
    medal = row.get("medal_earned")
    percentile = row.get("score_percentile")
    
    if medal:  # Medal-winning solution
        return "🥇 Medal"
    
    if percentile is None:
        return "🔴 Low"  # Default if no percentile
    
    if percentile >= 0.9:
        return "🔵 Very High"
    elif percentile >= 0.7:
        return "🟢 High"
    elif percentile >= 0.4:
        return "🟡 Medium"
    else:
        return "🔴 Low"


def build_summaries_df(summarization_folder: Path) -> pd.DataFrame:
    """Build SUMMARIES DataFrame from .md files in the summarization folder.
    
    Args:
        summarization_folder: Path to folder containing .md files with JSON summaries
        
    Returns:
        DataFrame with columns: rollout_id, score_percentile, medal_earned, tier,
        task_analysis, summary, data_preprocessing, feature_engineering,
        model_selection, training_methodology, evaluation_and_submission,
        notable_implementation_details, full_summary
    """
    def flatten_dict_to_str(d: dict | None) -> str:
        """Flatten a dict with subfields into a readable string.
        
        Example:
            {"cleaning": "...", "transformations": "..."}
            -> "cleaning: ...\ntransformations: ..."
        """
        if not d:
            return ""
        parts = []
        for key, value in d.items():
            if value is None:
                parts.append(f"{key}: null")
            elif isinstance(value, dict):
                # Recursively flatten nested dicts
                nested = flatten_dict_to_str(value)
                parts.append(f"{key}:\n  {nested.replace(chr(10), chr(10) + '  ')}")
            else:
                parts.append(f"{key}: {value}")
        return "\n".join(parts)
    
    md_files = list(summarization_folder.glob("*.md"))
    print(f"Parsing {len(md_files)} .md files...")
    
    summaries_data = []
    for md_file in md_files:
        try:
            with open(md_file) as f:
                content = f.read().strip()
            # Parse JSON content
            data = json.loads(content)
            
            # Extract fields from the JSON structure
            rollout_info = data.get("rollout_info", {})
            row = {
                "rollout_id": rollout_info.get("rollout_id"),
                "score_percentile": rollout_info.get("score_percentile"),
                "medal_earned": rollout_info.get("medal_earned"),
                "task_analysis": flatten_dict_to_str(data.get("task_analysis")),
                "summary": data.get("summary", ""),
                "data_preprocessing": flatten_dict_to_str(data.get("data_preprocessing")),
                "feature_engineering": flatten_dict_to_str(data.get("feature_engineering")),
                "model_selection": flatten_dict_to_str(data.get("model_selection")),
                "training_methodology": flatten_dict_to_str(data.get("training_methodology")),
                "evaluation_and_submission": flatten_dict_to_str(data.get("evaluation_and_submission")),
                "notable_implementation_details": data.get("notable_implementation_details"),
            }
            # Add tier assignment based on percentile and medal
            row["tier"] = assign_tier(row)
            # Add full_summary by assembling all parts
            row["full_summary"] = assemble_summary(row)
            summaries_data.append(row)
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse {md_file.name}: {e}")
        except Exception as e:
            print(f"Warning: Error processing {md_file.name}: {e}")
    
    df = pd.DataFrame(summaries_data)
    print(f"Built SUMMARIES DataFrame with {len(df)} rows")
    return df


# build_comparison_question() removed - no longer needed with multi-step pipeline


def create_rlm_instance(
    args: argparse.Namespace,
    setup_code: str,
    logger: RLMLogger,
    max_iterations: int = 10,
) -> RLM:
    """Create a fresh RLM instance with the given configuration."""
    return RLM(
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
        max_iterations=max_iterations,
        logger=logger,
        verbose=args.verbose,
    )


def run_analysis_step(
    rlm: RLM,
    step_name: str,
    root_prompt: str,
    prior_context: str = "",
) -> str:
    """Run a single analysis step and return the result.
    
    Args:
        rlm: RLM instance to use
        step_name: Name of the step for logging
        root_prompt: The focused prompt for this step
        prior_context: Output from previous steps to pass as context
        
    Returns:
        The response string from this step
    """
    print(f"\n{'=' * 80}")
    print(f"Running Step: {step_name}")
    print("=" * 80)
    
    full_prompt = f"{comparison_schema}\n\nQUESTION:\n{root_prompt}"
    
    result: RLMChatCompletion = rlm.completion(
        prompt=prior_context,
        root_prompt=full_prompt
    )
    
    print(f"[{step_name}] Completed")
    return result.response


def main() -> None:
    args = parse_args()

    # Validate required environment variables
    required_env_vars = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

    # =========================================================================
    # LOAD ROLLOUT SUMMARIZATION FOLDER
    # =========================================================================
    print("\n" + "=" * 80)
    print(f"Loading summarizations for task: {args.task_name}")
    print("=" * 80)
    
    summarization_folder = load_summarization_folder(args.summarization_dir, args.task_name)

    if summarization_folder is None:
        print("ERROR: Could not find summarization folder. Exiting.")
        return

    # =========================================================================
    # BUILD SUMMARIES DATAFRAME
    # =========================================================================
    summaries_df = build_summaries_df(summarization_folder)
    
    if summaries_df.empty:
        print("ERROR: No valid summaries found. Exiting.")
        return
    
    # Save to summarization folder for SETUP_CODE to load
    summaries_jsonl_path = summarization_folder / f"{args.model}_{args.job_name}_{args.run_id}_{args.task_name}_all_rollouts.jsonl"
    summaries_df.to_json(summaries_jsonl_path, orient="records", lines=True)
    print(f"Saved SUMMARIES to: {summaries_jsonl_path}")

    # =========================================================================
    # BUILD SETUP_CODE (shared across all steps)
    # =========================================================================
    setup_code_parts = [
        "import pandas as pd",
        "",
        "# Analysis prompts - available as variables for use in llm_query or llm_query_batched calls",
        f"MEDAL_DEEP_DIVE_PROMPT = {json.dumps(MEDAL_DEEP_DIVE_PROMPT)}",
        "",
        f"CRITICAL_FAILURE_PROMPT = {json.dumps(CRITICAL_FAILURE_PROMPT)}",
        "",
        f"# Load SUMMARIES DataFrame from jsonl",
        f"SUMMARIES = pd.read_json({json.dumps(str(summaries_jsonl_path))}, lines=True)",
        'print(f"Loaded SUMMARIES DataFrame with {len(SUMMARIES)} rows and columns: {list(SUMMARIES.columns)}")',
        "",
        "# Helper function injected via inspect.getsource()",
        inspect.getsource(assemble_summary),
        "",
        "# Pre-compute 'assembled' column so LLM can use df['assembled'] directly",
        "SUMMARIES['assembled'] = SUMMARIES.apply(lambda row: assemble_summary(row.to_dict()), axis=1)",
        "print(f'Pre-computed assembled column with {len(SUMMARIES)} entries')",
        "",
        "print('Helper function assemble_summary() loaded')",
    ]
    SETUP_CODE = "\n".join(setup_code_parts)

    # =========================================================================
    # MULTI-STEP ANALYSIS PIPELINE
    # =========================================================================
    print("\n" + "=" * 80)
    print("Starting multi-step comparison analysis...")
    print("=" * 80)

    task_suffix = f"_{args.task_name}" if args.task_name else ""
    step_results: dict[str, str] = {}
    
    # Define analysis steps: (step_name, prompt, max_iterations)
    analysis_steps = [
        ("2_medal_deep_dive", f"""## Task: Medal Solution Deep Dive

{MEDAL_DEEP_DIVE_PROMPT}

Use REPL to examine medal/very-high solutions in detail. Use `llm_query_batched` for parallel analysis if helpful.

When done, store your markdown analysis in a variable and call `FINAL_VAR(variable_name)`.
""", 10),
        ("3_critical_failure", f"""## Task: Critical Failure Analysis

{CRITICAL_FAILURE_PROMPT}

Use REPL to examine Low-tier solutions and identify failure patterns.

When done, store your markdown analysis in a variable and call `FINAL_VAR(variable_name)`.
""", 10),
        # Dimension-specific contrastive analysis (4a-4f)
        ("4a_data_preprocessing", DATA_PREPROCESSING_PROMPT, 8),
        ("4b_feature_engineering", FEATURE_ENGINEERING_PROMPT, 8),
        ("4c_model_selection", MODEL_SELECTION_PROMPT, 8),
        ("4d_training_methodology", TRAINING_METHODOLOGY_PROMPT, 8),
        ("4e_evaluation_submission", EVALUATION_SUBMISSION_PROMPT, 8),
        ("4f_implementation_details", IMPLEMENTATION_DETAILS_PROMPT, 8),
        # Final recommendations
        ("5_recommendations", RECOMMENDATIONS_PROMPT, 8),
    ]
    
    try:
        for step_name, step_prompt, step_max_iter in analysis_steps:
            # Create logger for this step
            step_logger = RLMLogger(
                log_dir=args.log_dir,
                file_name=f"{args.model}_{args.job_name}_{args.run_id}{task_suffix}_{step_name}"
            )
            
            # Create fresh RLM instance for each step
            rlm = create_rlm_instance(args, SETUP_CODE, step_logger, max_iterations=step_max_iter)
            
            # Build prior context from previous steps
            prior_context = ""
            if step_results:
                prior_sections = [f"## Prior Analysis: {k}\n{v}" for k, v in step_results.items()]
                prior_context = "\n\n---\n\n".join(prior_sections)
            
            # Run this step
            result = run_analysis_step(rlm, step_name, step_prompt, prior_context)
            step_results[step_name] = result
            
            # Save intermediate result
            step_md_path = Path(step_logger.log_file_path).with_suffix(".md")
            with open(step_md_path, "w") as f:
                f.write(result)
            print(f"Step result saved to: {step_md_path}")
            
    except Exception as e:
        print(f"ERROR: Analysis failed at step: {e}")
        sys.exit(1)

    # =========================================================================
    # COMBINE FINAL REPORT
    # =========================================================================
    print("\n" + "=" * 80)
    print("Combining final report...")
    print("=" * 80)
    
    final_report_parts = [
        f"# Contrastive Analysis Report: {args.task_name}",
        "",
        "---",
        "",
    ]
    
    section_titles = {
        "2_medal_deep_dive": "## 2. Medal Solution Deep Dive", 
        "3_critical_failure": "## 3. Critical Failure Analysis",
        "4a_data_preprocessing": "## 4a. Contrastive Analysis: Data Preprocessing",
        "4b_feature_engineering": "## 4b. Contrastive Analysis: Feature Engineering",
        "4c_model_selection": "## 4c. Contrastive Analysis: Model Selection",
        "4d_training_methodology": "## 4d. Contrastive Analysis: Training Methodology",
        "4e_evaluation_submission": "## 4e. Contrastive Analysis: Evaluation & Submission",
        "4f_implementation_details": "## 4f. Contrastive Analysis: Notable Implementation Details",
        "5_recommendations": "## 5. Actionable Recommendations",
    }
    
    for step_name, result in step_results.items():
        title = section_titles.get(step_name, f"## {step_name}")
        final_report_parts.append(title)
        final_report_parts.append("")
        final_report_parts.append(result)
        final_report_parts.append("")
        final_report_parts.append("---")
        final_report_parts.append("")
    
    final_report = "\n".join(final_report_parts)
    
    # Save combined report
    combined_md_path = Path(args.log_dir) / f"{args.model}_{args.job_name}_{args.run_id}{task_suffix}_combined.md"
    with open(combined_md_path, "w") as f:
        f.write(final_report)
    print(f"\nCombined report saved to: {combined_md_path}")


if __name__ == "__main__":
    main()