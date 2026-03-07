"""
Summarize MLE Bench rollout solutions using direct Azure OpenAI API calls.

This script:
1. Loads MLE Bench trajectory data from JSONL
2. Processes each rollout in parallel using Azure OpenAI to generate solution summaries
3. Parses JSON responses and converts to formatted markdown using assemble_summary
4. Saves summaries as timestamped .md files with error detection and auto-retry

Features:
- Skips already processed rollouts (unless previous run had errors)
- Auto-deletes and reprocesses failed summaries
- Parallel processing with configurable worker count
- JSON to markdown conversion with tier assignment
"""

import argparse
import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from lesson_utils import assemble_summary, assign_tier

load_dotenv()


# =============================================================================
# Configuration (defaults - can be overridden via CLI)
# =============================================================================
DEFAULT_CONFIG = {
    "run_id": "514",
    "model_name": "gpt-5",
    "job_name": "summarization",
    "log_dir": "/checkpoint/maui_sft/winnieyangwn/rlm_dumps/summarization/v6/",
    "account": "maui_sft",
    # Azure OpenAI settings
    "azure_host": "azure-services-fair-openai1-eastus2n3.azure-api.net",
    "azure_deployment": "gpt-5",
    "azure_api_version": "2025-03-01-preview",
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze MLE Bench rollout data with Azure OpenAI")
    parser.add_argument("--account", type=str, default=DEFAULT_CONFIG["account"], help="Account name for data path")
    parser.add_argument("--run_id", type=str, default=DEFAULT_CONFIG["run_id"], help="Run ID to analyze")
    parser.add_argument("--model", type=str, default=DEFAULT_CONFIG["model_name"], help="Model name to use")
    parser.add_argument("--job_name", type=str, default=DEFAULT_CONFIG["job_name"], help="Job name for logging")
    parser.add_argument("--task_name", type=str, required=True, help="Task name to analyze (required)")
    parser.add_argument("--log_dir", type=str, default=DEFAULT_CONFIG["log_dir"], help="Directory for log files")
    parser.add_argument("--max_workers", type=int, default=20, help="Max parallel workers for processing rollouts")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    # Azure OpenAI settings
    parser.add_argument("--azure_host", type=str, default=DEFAULT_CONFIG["azure_host"], help="Azure OpenAI host")
    parser.add_argument("--azure_deployment", type=str, default=DEFAULT_CONFIG["azure_deployment"], help="Azure OpenAI deployment")
    parser.add_argument("--azure_api_version", type=str, default=DEFAULT_CONFIG["azure_api_version"], help="Azure OpenAI API version")
    parser.add_argument("--azure_api_key", type=str, default=os.getenv("AZURE_OPENAI_API_KEY", "6524db61b4774663a00ba80558122ceb"), help="Azure OpenAI API key")
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
            if not content:
                return True
            if content.startswith("Error:"):
                return True
            return False
    except Exception:
        return True


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


def save_summary(summary: str, output_path: str, task_name: str) -> str | None:
    """Save summary as .md file after converting JSON to markdown format.
    
    Args:
        summary: The JSON summary string from API response
        output_path: Path to save the .md file (will add .md suffix if needed)
        task_name: Task name for the summary header
        
    Returns:
        The formatted markdown string, or None if empty/error
    """
    if not summary:
        print("No summary to save")
        return None
    
    if summary.startswith("Error:"):
        print(f"API call failed: {summary}")
        return None
    
    # Parse JSON response
    try:
        data = json.loads(summary)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response: {e}")
        # Save raw response as fallback with error prefix
        output_path_obj = Path(output_path)
        if output_path_obj.suffix != ".md":
            output_path_obj = output_path_obj.with_suffix(".md")
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path_obj, "w") as md_file:
            md_file.write(f"Error: JSON parse failed\n\n{summary}")
        return None
    
    # Extract rollout_info and compute tier
    rollout_info = data.get("rollout_info", {})
    tier = assign_tier(rollout_info)
    
    # Convert JSON to formatted markdown using assemble_summary
    formatted_markdown = assemble_summary(
        data=data,
        rollout_info=rollout_info,
        task_name=task_name,
        tier=tier,
        flatten_fn=flatten_dict_to_str,
    )
    
    output_path_obj = Path(output_path)
    if output_path_obj.suffix != ".md":
        output_path_obj = output_path_obj.with_suffix(".md")
    
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path_obj, "w") as md_file:
        md_file.write(formatted_markdown)
    print(f"Saved summary to {output_path_obj} ({len(formatted_markdown)} chars)")
    
    return formatted_markdown


# =============================================================================
# Summary Template (simplified - no REPL instructions needed)
# =============================================================================
SUMMARY_TEMPLATE = """## Task

Analyze the code solution for Agent Rollout #{rollout_idx} in the Kaggle competition **{task_name}**.

---

## Rollout Info

**Rollout ID:** {rollout_idx}  
**Score Percentile:** {percentile}
**Medal Earned:** {medal}

---

## Competition Description

{task_description}

---

## Code to Analyze
````python
{code}
````

---

## Field Guidance

**rollout info**
- `Rollout ID:` rollout_idx
- `Score Percentile:` percentile
- `Medal Earned:` medal (gold/silver/bronze/null if None)

**task_analysis**
- `goal`: A 1-2 sentence description of what the competition asks the model to predict or produce, the data modality, and the domain (e.g. "Predict house sale prices from tabular property features" or "Classify satellite images into 10 land-use categories").
- `task_type`: The output structure of the problem — e.g. binary classification, multi-class classification, multi-label classification, regression, ranking, object detection, semantic segmentation, sequence generation.
- `data_modality`: The primary input data type — tabular, image, text, time-series, audio, graph, or multimodal (specify which combination).
- `evaluation_metric`: The exact metric used to score submissions (e.g. RMSE, macro F1, mAP@0.5, log loss). Include whether higher or lower is better, and any nuances in how it is computed (e.g. sample weighting, per-class averaging).
- `core_challenges`: The key technical difficulties inherent to this competition — e.g. severe class imbalance, high-cardinality categoricals, noisy labels, limited training data, multimodal inputs, long-tailed distributions, covariate shift between train and test etc.
- `difficulty_factors`: Factors that make this competition hard to solve well — e.g. large data scale, complex feature interactions, domain expertise required etc. Distinguish from core challenges: challenges are about the data/task structure, difficulty factors are about what makes it hard to get a good solution.

**summary**: A 2-3 sentence overview of the solution — what kind of model was used, any standout techniques, and the overall approach. Should be readable as a standalone description of the solution.

**data_preprocessing**
- `cleaning`: Handling of missing values (imputation, dropping), outlier removal, filtering of invalid rows/columns, and class imbalance handling (oversampling, undersampling, SMOTE, class weights).
- `transformations`: Scaling (StandardScaler, MinMax), encoding (one-hot, label, target), type casting, reshaping, tokenization. 
- `pipeline_structure`: How transformations are fit and applied — whether fitted on train only or the full dataset, use of sklearn Pipeline or equivalent, and how the same transforms are applied at inference. 
- `augmentation`: Image/text/tabular augmentation techniques applied (e.g. flips, cutmix, synonym replacement).

**feature_engineering**
- `new_features`: Derived columns and how they were computed (e.g. interaction terms, lag features, TF-IDF, embeddings). Note "none" explicitly if absent.
- `selection_or_reduction`: Feature selection methods (e.g. importance-based pruning, correlation filtering) or dimensionality reduction (PCA, UMAP).

**model_selection**:
- `algorithms`: Primary algorithm(s) used — exact class/function name (e.g. `LGBMClassifier`, `ResNet50`). Note if only a trivial baseline was used (e.g. `DummyClassifier`, majority-class prediction).
- `hyperparameters`: Key hyperparameter values and whether they were tuned, set to defaults, or hardcoded.
- `ensemble`: Ensemble architecture if applicable (stacking, blending, voting), number of models, and how predictions were combined. Null if none.
- `pretrained_models`: Pretrained models used and how (feature extraction vs. fine-tuning). Null if none.

**training_methodology**:
- `objective_alignment`: Whether the training objective matches the competition evaluation metric. Flag mismatches explicitly.
- `validation_strategy`: How the agent validated performance — k-fold, stratified fold, holdout split, or no validation. Note if validation was absent or potentially leaky.
- `training_configuration`: Epochs, batch size, optimizer, scheduler, loss function. For tree models: n_estimators, early stopping rounds, learning rate.
- `tuning`: Hyperparameter tuning approach — grid search, Optuna, manual, or none.

**evaluation_and_submission**:
- `prediction_method`: How final predictions were produced (mean, median, weighted average, rank averaging, argmax, threshold).
- `post_processing`: Any post-processing applied (clipping, calibration, threshold tuning). Null if none.

**notable_implementation_details**: Any computational considerations (GPU/CPU usage, mixed precision, runtime tricks), unique or novel techniques not covered above, and other significant aspects of the solution.

---

## Instructions

Analyze the solution and respond with a JSON object only — no markdown, no explanation, no surrounding text.

Your summary will be read by researchers who do not have access to the original code. Write as if you are the sole source of truth about this solution. This means:

- **Be factual**: Only describe what is explicitly present in the code. Do not infer intent, assume standard practices, or fill gaps with what "typically" happens. If something is absent or unclear, say so explicitly.
- **Be specific**: Reference actual class names, function calls, variable names, column names, and parameter values from the code (e.g. `LGBMClassifier(n_estimators=500, learning_rate=0.05)`, not "a gradient boosting model with tuned hyperparameters").
- **Be concise**: Be thorough but economical; omit details that don't change how a reader would understand the solution.
- **Be complete**: A reader with no access to the code should be able to fully reconstruct the logical flow of the solution — from how data enters the pipeline to how the final submission file is produced — using only your summary.
- **Null fields**: Use `null` only when something is genuinely absent from the code — not when it is unclear or underdescribed. If partially present, describe what is there.

The JSON must follow this schema exactly:
````json
{{
  "rollout_info": {{
    "rollout_id": <int>,    
    "score_percentile": <float>,
    "medal_earned": <string or null>
  }},
  "task_analysis": {{
  "goal": "<string>",
  "task_type": "<string>",
  "data_modality": "<string>",
  "evaluation_metric": "<string>",
  "core_challenges": "<string>",
  "difficulty_factors": "<string>"
 }},
  "summary": "<string>",
  "data_preprocessing": {{
  "cleaning": "<string>",
  "transformations": "<string>",
  "pipeline_structure": "<string>",
  "augmentation": "<string or null>"
  }},
  "feature_engineering": {{
    "new_features": "<string or null>",
    "selection_or_reduction": "<string or null>"
  }},
  "model_selection": {{
  "algorithms": "<string>",
  "hyperparameters": "<string>",
  "ensemble": "<string or null>",
  "pretrained_models": "<string or null>"
  }},
  "training_methodology": {{
  "objective_alignment": "<string>",
  "validation_strategy": "<string>",
  "training_configuration": "<string>",
  "tuning": "<string or null>"
 }},
  "evaluation_and_submission": {{
  "prediction_method": "<string>",
  "post_processing": "<string or null>"
  }},ls
  
  "notable_implementation_details": "<string or null>"
}}
````"""

def build_prompt(task_name: str, rollout_idx: int, percentile: float | None, medal: str | None, code: str, task_description: str) -> str:
    """Build the analysis prompt for a single rollout."""
    return SUMMARY_TEMPLATE.format(
        task_name=task_name,
        rollout_idx=rollout_idx,
        percentile=percentile,
        medal=medal or 'none',
        code=code,
        task_description=task_description,
    )


def call_azure_openai(
    prompt: str,
    host: str,
    deployment: str,
    api_key: str,
    api_version: str,
) -> str:
    """Call Azure OpenAI API directly.
    
    Args:
        prompt: The user prompt
        host: Azure OpenAI host
        deployment: Azure deployment name
        api_key: API key
        api_version: API version
        
    Returns:
        The model's response text, or error message
    """
    url = f"https://{host}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key,
    }
    payload = {
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=300)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except requests.exceptions.Timeout:
        return "Error: Request timed out"
    except requests.exceptions.RequestException as e:
        return f"Error: Request failed - {e}"
    except (KeyError, IndexError) as e:
        return f"Error: Failed to parse response - {e}"


def generate_output_filename(args: argparse.Namespace, task_name: str, rollout_idx: int) -> str:
    """Generate a timestamped output filename."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    short_uuid = str(uuid.uuid4())[:8]
    return f"{args.model}_{args.job_name}_{args.run_id}_{task_name}_rollout{rollout_idx}_{timestamp}_{short_uuid}.md"


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
    code = row.get("code")
    task_description = row.get("task_description", "")
    
    # Skip if no code
    if not code:
        print(f"  Skipping rollout {rollout_idx}: no code submitted")
        return None
    
    print(f"\n{'='*60}")
    print(f"Processing Rollout {rollout_idx}: {task_name}")
    print(f"  Percentile: {percentile}, Medal: {medal}")
    print(f"{'='*60}")
    
    # Build prompt
    prompt = build_prompt(task_name, rollout_idx, percentile, medal, code, task_description)
    
    if args.verbose:
        print(f"Prompt length: {len(prompt)} chars")
    
    # Call Azure OpenAI
    print(f"Calling Azure OpenAI ({args.azure_deployment})...")
    response = call_azure_openai(
        prompt=prompt,
        host=args.azure_host,
        deployment=args.azure_deployment,
        api_key=args.azure_api_key,
        api_version=args.azure_api_version,
    )
    
    # Generate output filename and save
    output_filename = generate_output_filename(args, task_name, rollout_idx)
    output_path = output_dir / output_filename
    summary = save_summary(response, str(output_path), task_name)
    
    return summary


def main() -> None:
    args = parse_args()
    
    # Build paths
    data_path = get_data_path(args.account, args.run_id)
    validate_path(data_path, "Data file")
    
    # Load data
    print(f"Loading data from {data_path}...")
    all_rollouts_df = pd.read_json(data_path, lines=True)
    print(f"Loaded {len(all_rollouts_df)} total rollouts")
    
    # Filter by task_name
    all_rollouts_df = all_rollouts_df[all_rollouts_df['task_name'] == args.task_name]
    print(f"Filtered to {len(all_rollouts_df)} rollouts for task: {args.task_name}")
    
    # Filter to only valid submissions
    all_rollouts_df = all_rollouts_df[all_rollouts_df['valid_submission'] == True]
    print(f"Filtered to {len(all_rollouts_df)} valid submissions")
    
    if len(all_rollouts_df) == 0:
        print("No rollouts found matching criteria. Exiting.")
        return
    
    # Create output directory
    output_dir = Path(args.log_dir) / args.task_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Track results
    successful = []
    failed = []
    skipped = []
    
    # Build list of rollouts to process
    rollouts_to_process = []
    for df_idx, row in all_rollouts_df.iterrows():
        row_dict = row.to_dict()
        task_name = row_dict.get("task_name", "unknown")
        md_pattern = f"{args.model}_{args.job_name}_{args.run_id}_{task_name}_rollout{df_idx}_*.md"
        existing_md_files = list(output_dir.glob(md_pattern))
        if existing_md_files:
            existing_md = max(existing_md_files, key=lambda p: p.stat().st_mtime)
            if has_error_in_md(existing_md):
                print(f"  Reprocessing rollout {df_idx}: previous run had error ({existing_md.name})")
                for md_file in existing_md_files:
                    md_file.unlink()
                    print(f"    Deleted: {md_file.name}")
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
