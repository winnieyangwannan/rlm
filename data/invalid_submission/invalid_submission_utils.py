import matplotlib.pyplot as plt
import pandas as pd
try:
    from IPython.display import display
except ImportError:
    display = None


# Define infrastructure error categories (shared across all functions)
INFRA_ERROR_CATEGORIES = {
    "Infrastructure: Connection/Heartbeat Error",
    "Infrastructure: AgentBox Backend Not Initialized",
    "Infrastructure: Container Error",
    "Infrastructure: Container Execution Error",
    "Infrastructure: Container Creation Error",
    "Infrastructure: Model Call Error",
    "Too many streaming blocks of output",
}


def is_infra_error_category(category: str) -> bool:
    """Check if an error category is an infrastructure error."""
    return category in INFRA_ERROR_CATEGORIES


def categorize_error(row):
    """Categorize error based on common patterns in the error message and infrastructure flags.
    
    Args:
        row: A dict-like object containing error_output and infrastructure flags.
              Expected keys: eval_error_output, info_output, rollout_str, parse_error,
              container_execution_error, container_creation_error, model_call_error, max_turns_reached
    """
    error_output = row.get("eval_error_output", "")
    error_str = str(error_output).lower()
    error_str_original = str(error_output)  # Keep original case for some checks
    info_output = row.get("info_output", "")
    rollout_str = row.get("rollout_str", "")
    parse_error = row.get("parse_error", False)
    container_execution_error = row.get("container_execution_error", False)
    container_creation_error = row.get("container_creation_error", False)
    model_call_error = row.get("model_call_error", False)
    max_turns_reached = row.get("max_turns_reached", False)
    rollout_timeout = row.get("rollout_timeout", False)
    info_str = str(info_output).lower() if info_output else ""

    # Check infrastructure errors FIRST (these prevent evaluation from running)
    # When container_execution_error=True, evaluation was never called
    if container_execution_error:
        # Check info_output for specific infrastructure error details
        if "worker connection failed" in info_str or "heartbeat" in info_str or "socket closed" in info_str or "grpc" in info_str:
            return "Infrastructure: Connection/Heartbeat Error"
        elif "agentbox container error" in info_str:
            return "Infrastructure: Container Error"
        elif "'agentboxbackend' object has no attribute 'container'" in str(rollout_str).lower():
            return "Infrastructure: AgentBox Backend Not Initialized"
        elif "agentboxbackend" in info_str and "has no attribute" in info_str and "container" in info_str:
            return "Infrastructure: AgentBox Backend Not Initialized"
        else:
            return "Infrastructure: Container Execution Error"
    
    elif container_creation_error:
        return "Infrastructure: Container Creation Error"
    elif "agentboxbackend" in info_str and "has no attribute" in info_str and "container" in info_str:
            return "Infrastructure: AgentBox Backend Not Initialized"
    elif "agentboxbackend" in error_str and "has no attribute" in error_str and "container" in error_str:
            return "Infrastructure: AgentBox Backend Not Initialized"
    elif "'agentboxbackend' object has no attribute 'container'" in str(rollout_str).lower():
            return "Infrastructure: AgentBox Backend Not Initialized"
    elif model_call_error:
        return "Infrastructure: Model Call Error"
    
    elif max_turns_reached:
        return "Max Turns Reached"
    elif parse_error:
        return "Tool Call Parsing Error"
    else:
        if error_str.strip() == "" or error_str == "none":
            # Check infrastructure flags for empty error messages
            if container_execution_error:
                return "Infrastructure: Container Execution Error"
            if container_creation_error:
                return "Infrastructure: Container Creation Error"
            if rollout_timeout:
                return "Rollout Timeout"
            # Check rollout_str for AgentBoxBackend error when error message is empty
            rollout_str_lower = str(rollout_str).lower()
            if "'agentboxbackend' object has no attribute 'container'" in rollout_str_lower:
                return "Infrastructure: AgentBox Backend Not Initialized"
            # If max_turns_reached and error is empty, categorize as "Max Turns Reached"
            if max_turns_reached:
                return "Max Turns Reached"
            return "Empty error message"

    
    # Check for common error patterns - ORDER MATTERS (more specific patterns first)
    
    # Worker/Infrastructure issues (check early as they're specific)
    if "worker connection failed" in error_str or "worker became unresponsive" in error_str:
        return "Infrastructure: Connection/Heartbeat Error"
    elif "heartbeat" in error_str and ("fail" in error_str or "timeout" in error_str):
        return "Infrastructure: Connection/Heartbeat Error"
    elif "socket closed" in error_str or "grpc" in error_str:
        return "Infrastructure: Connection/Heartbeat Error"
    
    # OOM Kill - check for "Killed" message (case-sensitive, usually appears as "Killed\n")
    # This happens when Linux OOM killer terminates the process
    # Also catches DataLoader worker killed messages like "is killed by signal: Killed"
    if "Killed" in error_str_original and ("killed\n" in error_str or error_str.strip().endswith("killed")):
        return "OOM Killed (Linux OOM Killer)"
    elif error_str.strip() == "killed" or error_str.strip().startswith("killed\n"):
        return "OOM Killed (Linux OOM Killer)"
    elif "is killed by signal" in error_str or "killed by signal: killed" in error_str:
        return "OOM Killed (Linux OOM Killer)"
    
    # Incomplete execution patterns (training/caching stopped mid-way)
    if "caching images" in error_str and "ram" in error_str:
        return "Incomplete Execution (Caching Images)"
    elif ("training" in error_str or "epoch" in error_str) and not any(err in error_str for err in ["error", "exception", "traceback"]):
        # Training output without error - likely incomplete
        if "100%" not in error_str:  # Not completed
            return "Incomplete Execution (Training Stopped)"
    
    # Environment/Dependency errors (numpy version conflicts, pip issues)
    if "cannot convert numpy.ndarray to numpy.ndarray" in error_str:
        return "Environment/Dependency Error"
    elif "pip's dependency resolver" in error_str and ("numpy<2.0" in error_str or "numpy<1." in error_str):
        return "Environment/Dependency Error"
    elif "which is incompatible" in error_str and ("numpy" in error_str or "cuda-python" in error_str):
        return "Environment/Dependency Error"
    
    # Validation errors (submission format issues from grading)
    if "validation error" in error_str and "submission invalid" in error_str:
        return "Submission Validation Error (Grading)"
    elif "the set of" in error_str and "must match" in error_str:
        return "Submission ID Mismatch"
    if "submission.csv not in solution" in error_str:
        return "Missing submission.csv in solution"
    # Check for specific submission.csv not found error
    if "/workspace/submission.csv: no such file or directory" in error_str:
        return "csv_not_found"
    
    # Standard error patterns
    if "no solution found" in error_str or "no such file" in error_str:
        return "SolutionNotFoundError"
    if "filenotfounderror" in error_str or "no such file" in error_str:
        return "FileNotFoundError"
    elif "error tokenizing data" in error_str and "c error:" in error_str:
        return "CSV Tokenization Error"
    elif "too many streaming blocks" in error_str: 
        return "Infrastructure: Too many streaming blocks of output"
    elif "timeout" in error_str:
        return "Timeout"
    elif "submission.csv" in error_str and ("not found" in error_str or "missing" in error_str or "does not exist" in error_str):
        return "Missing submission.csv"
    elif "submission and answers should have" in error_str:
        return "Wrong Submission Format"
    elif "submission and answers have different lengths" in error_str:
        return "Submission/Answers Length Mismatch"
    elif "memoryerror" in error_str or "out of memory" in error_str or "oom" in error_str:
        return "MemoryError"
    elif "valueerror" in error_str:
        return "ValueError"
    elif "systemexit: 2" in error_str or "systemexit(2)" in error_str or ("argparse" in error_str and "error" in error_str):
        return "ArgparseError (SystemExit: 2)"
    elif "systemexit: 1" in error_str or "systemexit(1)" in error_str:
        return "General Script Error (SystemExit: 1)"
    elif "systemexit: 126" in error_str or "systemexit(126)" in error_str:
        return "Command Cannot Execute (SystemExit: 126)"
    elif "systemexit: 127" in error_str or "systemexit(127)" in error_str:
        return "Command Not Found (SystemExit: 127)"
    elif "systemexit: 130" in error_str or "systemexit(130)" in error_str:
        return "Interrupted (Ctrl+C) (SystemExit: 130)"
    elif "systemexit: 137" in error_str or "systemexit(137)" in error_str:
        return "Killed (SIGKILL) (SystemExit: 137)"
    elif "systemexit: 139" in error_str or "systemexit(139)" in error_str:
        return "Segmentation Fault (SystemExit: 139)"
    elif "systemexit: 143" in error_str or "systemexit(143)" in error_str:
        return "Terminated (SIGTERM) (SystemExit: 143)"
    elif "systemexit:" in error_str or "systemexit(" in error_str:
        return "Other SystemExit"
    elif "syntaxerror" in error_str:
        return "SyntaxError"
    elif "importerror" in error_str or "modulenotfounderror" in error_str:
        return "ImportError"
    elif "keyerror" in error_str:
        return "KeyError"
    elif "typeerror" in error_str:
        return "TypeError"
    elif "indexerror" in error_str:
        return "IndexError"
    elif "attributeerror" in error_str:
        return "AttributeError"
    elif "runtimeerror" in error_str:
        return "RuntimeError"
    elif "zerodivisionerror" in error_str:
        return "ZeroDivisionError"
    elif "permissionerror" in error_str or "permission denied" in error_str:
        return "PermissionError"
    elif "connectionerror" in error_str or "connection refused" in error_str:
        return "ConnectionError"
    elif "shape" in error_str and ("mismatch" in error_str or "different" in error_str):
        return "Shape Mismatch"
    elif "column" in error_str and ("missing" in error_str or "not found" in error_str):
        return "Missing Column"

    elif "invalid" in error_str and "format" in error_str:
        return "Invalid Format"

    else:
        return "Other"


def plot_submission_validity_breakdown(df, figsize=(5, 3), plot=True):
    """
    Create a bar chart showing the breakdown of valid vs invalid vs infrastructure error submissions.
    
    Args:
        df: DataFrame containing 'valid_submission' and optionally 'container_execution_error' column
        figsize: Tuple specifying figure size (width, height)
        plot: Whether to display the plot. Default is True. Set to False to skip plotting.
        
    Returns:
        tuple: (filtered_valid, filtered_invalid, filtered_infra) DataFrames for further analysis
    """
    # Use the shared categorize_error function to detect infrastructure errors
    def get_error_category(row):
        """Get error category for a row using the shared categorize_error function."""
        return categorize_error({
            "eval_error_output": row.get("eval_error_output", ""),
            "info_output": "",  # Not available in flattened df
            "rollout_str": row.get("rollout_str", ""),
            "parse_error": False,
            "container_execution_error": row.get("container_execution_error", False),
            "container_creation_error": row.get("container_creation_error", False),
            "model_call_error": row.get("model_call_error", False),
            "max_turns_reached": row.get("max_turns_reached", False),
            "rollout_timeout": row.get("rollout_timeout", False),
        })
    
    # Categorize each row and check if it's an infrastructure error
    error_categories = df.apply(get_error_category, axis=1)
    infra_mask = error_categories.apply(is_infra_error_category)
    
    # Filter data into 3 categories
    filtered_infra = df[infra_mask]
    filtered_valid = df[(df['valid_submission'] == True) & (~infra_mask)]
    filtered_invalid = df[(df['valid_submission'] == False) & (~infra_mask)]
    
    print(f"Found {len(filtered_valid)} rollouts matching valid criteria")
    print(f"Found {len(filtered_invalid)} rollouts matching invalid criteria (excluding infra errors)")
    print(f"Found {len(filtered_infra)} rollouts with infrastructure errors")

    # If there are infrastructure errors, print a breakdown table by category
    if len(filtered_infra) > 0:
        # Use the already computed error categories
        infra_categories = error_categories[infra_mask]
        infra_category_counts = infra_categories.value_counts().reset_index()
        infra_category_counts.columns = ['Infrastructure Error Category', 'Count']
        infra_category_counts['Percentage'] = (infra_category_counts['Count'] / len(filtered_infra) * 100).round(1)
        infra_category_counts = infra_category_counts.sort_values('Count', ascending=False).reset_index(drop=True)
        
        print(f"\nInfrastructure Error Breakdown:")
        try:
            from IPython.display import display
            display(infra_category_counts)
        except ImportError:
            print(infra_category_counts.to_string(index=False))

    # Prepare data
    valid_count = len(filtered_valid)
    invalid_count = len(filtered_invalid)
    infra_count = len(filtered_infra)
    total_count = valid_count + invalid_count + infra_count
    eval_total = valid_count + invalid_count

    # Always create plot with 3 bars
    if plot:
        fig, ax = plt.subplots(figsize=figsize)
        categories = ['Valid', 'Invalid\n(Eval Errors)', 'Infrastructure\nErrors']
        counts = [valid_count, invalid_count, infra_count]
        colors = ['#2ecc71', '#e74c3c', '#f39c12']  # Green, Red, Orange
        bars = ax.bar(categories, counts, color=colors)

        percentages = [
            valid_count / eval_total * 100 if eval_total > 0 else 0,
            invalid_count / eval_total * 100 if eval_total > 0 else 0,
            infra_count / total_count * 100 if total_count > 0 else 0
        ]
        for bar, count, pct in zip(bars, counts, percentages):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{count}\n({pct:.1f}%)', 
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_ylabel('Number of Submissions', fontsize=12)
        ax.set_xlabel('Submission Status', fontsize=12)
        ax.set_title('Submission Breakdown: Valid vs Invalid vs Infrastructure Errors', fontsize=12, fontweight='bold')
        ax.set_ylim(0, max(counts) * 1.2 if max(counts) > 0 else 1)

        plt.tight_layout()
        plt.show()
    
    return filtered_valid, filtered_invalid, filtered_infra


def analyze_invalid_submissions(df):
    """
    Comprehensive analysis of invalid submissions including error extraction, 
    categorization, and summary reporting.
    
    Args:
        df: Flattened DataFrame with columns including 'valid_submission', 'eval_error_output',
            'task_name', 'container_execution_error', and 'row_indx'.
            This is the output of flatten_dataframe().
        
    Returns:
        tuple: (df_errors, summary_df, rollout_indices_df) for further analysis
               The 'index' column in df_errors and 'Rollout Indices' in rollout_indices_df
               are row indices (iloc positions) of the input df.
    """
    import numpy as np
    
    def extract_error_info(df):
        """Extract error information from invalid submissions in flattened dataframe."""
        # Filter to only invalid submissions
        invalid_mask = df['valid_submission'] == False
        invalid_df = df[invalid_mask].copy()
        
        # Get integer positions (iloc) of invalid rows in original df
        invalid_iloc_positions = np.where(invalid_mask)[0]
        
        # Build the errors dataframe from flat columns
        eval_errors = []
        for iloc_pos, (idx, row) in zip(invalid_iloc_positions, invalid_df.iterrows()):
            eval_errors.append({
                "index": iloc_pos,  # Row index (iloc position) in original df
                "task_name": row.get("task_name", "unknown"),
                "eval_error_output": row.get("eval_error_output", ""),
                "eval_error_message": row.get("eval_error_message", ""),
                "pred_solution_provided": row.get("pred_solution_provided", None),
                "eval_outcome": row.get("eval_outcome", None),
                "eval_exception_type": row.get("eval_exception_type", None),
                "container_execution_error": row.get("container_execution_error", False),
                "container_creation_error": row.get("container_creation_error", False),
                "model_call_error": row.get("model_call_error", False),
                "max_turns_reached": row.get("max_turns_reached", False),
                "rollout_timeout": row.get("rollout_timeout", False),
                "info_output": row.get("info_output", ""),
                "parse_error": row.get("parse_error", False),
                "rollout_str": row.get("rollout_str", ""),
            })

        return pd.DataFrame(eval_errors)
    
    # Extract error information
    print(f"Analyzing {len(df)} rows for invalid submissions...")
    df_errors = extract_error_info(df)
    print(f"Found {len(df_errors)} invalid submissions")
    
    # Apply categorization (now uses full row for infrastructure error detection)
    df_errors["error_category"] = df_errors.apply(categorize_error, axis=1)

    # Display error category distribution as a table
    error_counts = df_errors["error_category"].value_counts()
    error_pct = (error_counts / error_counts.sum() * 100).round(2)

    # Create summary table
    summary_df = pd.DataFrame({
        "Count": error_counts,
        "Percentage": error_pct
    })
    summary_df.index.name = "Error Category"
    print("\n" + "="*60)
    print("ERROR CATEGORY DISTRIBUTION")
    print("="*60)
    display(summary_df)
    print(f"\nTotal: {error_counts.sum()}")

    # Create table with rollout indices for each category
    rollout_indices_data = []
    for category in error_counts.index:
        category_df = df_errors[df_errors["error_category"] == category]
        indices = category_df["index"].tolist()
        rollout_indices_data.append({
            "Error Category": category,
            "Count": len(indices),
            "Rollout Indices": str(indices)
        })

    rollout_indices_df = pd.DataFrame(rollout_indices_data)
    print("\n" + "="*60)
    print("ROLLOUT INDICES BY ERROR CATEGORY")
    print("="*60)
    display(rollout_indices_df)
    
    return df_errors, summary_df, rollout_indices_df


def plot_invalid_error_distribution(df_errors, figsize=(12, 6), task_names=None):
    """
    Visualize the distribution of error types in invalid submissions.
    
    Args:
        df_errors: DataFrame with 'error_category' column containing categorized errors
        figsize: Tuple specifying figure size (width, height)
        task_names: Optional list of task names to filter on. If None, use all tasks.
        
    Returns:
        tuple: (error_counts, error_pct) for further analysis
    """
    import matplotlib.pyplot as plt
    
    # Filter by task_names if provided
    if task_names is not None:
        df_errors = df_errors[df_errors['task_name'].isin(task_names)]
        print(f"Filtered to {len(df_errors)} errors for {len(task_names)} tasks")
    
    # Calculate error distribution
    error_counts = df_errors["error_category"].value_counts()
    error_pct = (error_counts / error_counts.sum() * 100).round(2)
    
    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=figsize)
    bars = error_counts.plot(kind='barh', ax=ax, color='steelblue')
    ax.set_xlabel('Count')
    ax.set_ylabel('Error Category')
    
    # Set title based on whether task_names is provided
    if task_names is not None:
        if len(task_names) == 1:
            title = f'Distribution of Error Types in Invalid Submissions\nTask: {task_names[0]}'
        else:
            title = f'Distribution of Error Types in Invalid Submissions\n({len(task_names)} tasks)'
    else:
        title = 'Distribution of Error Types in Invalid Submissions'
    ax.set_title(title)

    # Add percentage labels on each bar
    total = error_counts.sum()
    for i, (count, category) in enumerate(zip(error_counts.values, error_counts.index)):
        pct = count / total * 100
        ax.text(count + 0.5, i, f'{count} ({pct:.1f}%)', va='center', fontsize=9)

    # Adjust x-axis to make room for labels
    ax.set_xlim(0, error_counts.max() * 1.25)

    plt.tight_layout()
    plt.show()

    # Show percentage breakdown
    print("\n" + "=" * 60)
    print("ERROR CATEGORY PERCENTAGE")
    print("=" * 60)
    for cat, pct in error_pct.items():
        print(f"{cat}: {pct}%")
    
    return error_counts, error_pct


def investigate_error_type(df_errors, df_trj, error_type="Other", sample_idx=0, show_full_output=False):
    """
    Investigate a specific error type from invalid submissions.
    
    Parameters:
    -----------
    df_errors : pd.DataFrame
        DataFrame containing error information from analyze_invalid_submissions()
    df_trj : pd.DataFrame
        Original trajectory DataFrame
    error_type : str
        Type of error to investigate. Options typically include:
        "Other", "Timeout", "Submission Not Found", "Invalid Submission Format", etc.
    sample_idx : int
        Index of the sample to investigate within the filtered errors (default: 0)
    show_full_output : bool
        If True, prints the full error output and pred_solution (default: False)
    
    Returns:
    --------
    dict containing:
        - 'filtered_errors': DataFrame of errors matching the error_type
        - 'error_counts': Count of errors by task
        - 'eval_error_output': The error output for the selected sample
        - 'pred_solution': The predicted solution for the selected sample
        - 'task_name': The task name for the selected sample
        - 'original_idx': The index in df_trj for the selected sample
    """
    # Get filtered errors and counts
    filtered_errors, error_counts = analyze_error_by_task(df_errors, error_type)
    
    if len(filtered_errors) == 0:
        print(f"No errors found for error type: '{error_type}'")
        print(f"Available error types: {df_errors['error_category'].unique().tolist()}")
        return {
            'filtered_errors': filtered_errors,
            'error_counts': error_counts,
            'eval_error_output': None,
            'pred_solution': None,
            'task_name': None,
            'original_idx': None
        }
    
    if sample_idx >= len(filtered_errors):
        print(f"sample_idx {sample_idx} is out of range. Max index: {len(filtered_errors) - 1}")
        sample_idx = 0
    
    # Get the original index in df_trj from the 'index' column (NOT the DataFrame row position)
    original_idx = filtered_errors.iloc[sample_idx]["index"]
    
    # Get error output and pred_solution
    eval_error_output = filtered_errors.iloc[sample_idx]["eval_error_output"]
    pred_solution = df_trj.iloc[original_idx]["rollouts"][0]["traj"]["transitions"][-1]["info"].get("pred_solution", None)
    task_name = filtered_errors.iloc[sample_idx].get("task_name", "Unknown")
    
    print(f"Error Type: {error_type}")
    print(f"Total samples with this error: {len(filtered_errors)}")
    print(f"Investigating sample {sample_idx} (original index: {original_idx})")
    print(f"Task: {task_name}")
    print("-" * 50)
    
    if show_full_output:
        print("\n=== Eval Error Output ===")
        print(eval_error_output)
        print("\n=== Predicted Solution ===")
        print(pred_solution)
    
    return {
        'filtered_errors': filtered_errors,
        'error_counts': error_counts,
        'eval_error_output': eval_error_output,
        'pred_solution': pred_solution,
        'task_name': task_name,
        'original_idx': original_idx
    }


def analyze_error_by_task(df_errors, error_category, figsize=(10, 6), top_n=15):
    """
    Analyze specific error category by task distribution.
    
    Args:
        df_errors: DataFrame with error categorization
        error_category: String specifying which error category to analyze
        figsize: Tuple specifying figure size for visualization 
        top_n: Number of top tasks to display in visualization
        
    Returns:
        tuple: (filtered_errors, task_counts) for further analysis
    """
    import matplotlib.pyplot as plt
    
    # Filter for specified error category
    filtered_errors = df_errors[df_errors["error_category"] == error_category]
    
    # Task distribution for specified errors
    task_counts = filtered_errors["task_name"].value_counts()
    
    # Print summary
    print("=" * 80)
    print(f"TASKS AFFECTED BY '{error_category.upper()}' ERROR")
    print("=" * 80)
    print(f"Total: {len(filtered_errors)} cases across {len(task_counts)} unique tasks\n")

    # Show all tasks if any exist
    if len(task_counts) > 0:
        print(task_counts.to_string())

        # Visualize top N tasks
        fig, ax = plt.subplots(figsize=figsize)
        top_tasks = task_counts.head(top_n)
        top_tasks.plot(kind='barh', ax=ax, color='coral')
        ax.set_xlabel('Count')
        ax.set_ylabel('Task')
        ax.set_title(f'Top {min(top_n, len(task_counts))} Tasks with "{error_category}" Error', fontweight='bold')

        # Add percentage labels on each bar (percentage of all errors of this type)
        total_errors = len(filtered_errors)
        for i, (count, task) in enumerate(zip(top_tasks.values, top_tasks.index)):
            pct = count / total_errors * 100
            ax.text(count + 0.2, i, f'{count} ({pct:.1f}%)', va='center', fontsize=9)

        # Adjust x-axis to make room for labels
        ax.set_xlim(0, top_tasks.max() * 1.3)

        plt.tight_layout()
        plt.show()
        
        print(f"\n(Percentages are calculated with respect to all {total_errors} '{error_category}' errors)")
    else:
        print(f"No '{error_category}' errors found in the data.")
    
    return filtered_errors, task_counts


def investigate_error_type(df_errors, df_trj, error_type="Other", sample_idx=0, show_full_output=False):
    """
    Investigate a specific error type from invalid submissions.
    
    Parameters:
    -----------
    df_errors : pd.DataFrame
        DataFrame containing error information from analyze_invalid_submissions()
    df_trj : pd.DataFrame
        Original trajectory DataFrame
    error_type : str
        Type of error to investigate. Options typically include:
        "Other", "Timeout", "Submission Not Found", "Invalid Submission Format", etc.
    sample_idx : int
        Index of the sample to investigate within the filtered errors (default: 0)
    show_full_output : bool
        If True, prints the full error output and pred_solution (default: False)
    
    Returns:
    --------
    dict containing:
        - 'filtered_errors': DataFrame of errors matching the error_type
        - 'error_counts': Count of errors by task
        - 'eval_error_output': The error output for the selected sample
        - 'pred_solution': The predicted solution for the selected sample
        - 'task_name': The task name for the selected sample
        - 'original_idx': The index in df_trj for the selected sample
    """
    # Get filtered errors and counts
    filtered_errors, error_counts = analyze_error_by_task(df_errors, error_type)
    
    if len(filtered_errors) == 0:
        print(f"No errors found for error type: '{error_type}'")
        print(f"Available error types: {df_errors['error_category'].unique().tolist()}")
        return {
            'filtered_errors': filtered_errors,
            'error_counts': error_counts,
            'eval_error_output': None,
            'pred_solution': None,
            'task_name': None,
            'original_idx': None
        }
    
    if sample_idx >= len(filtered_errors):
        print(f"sample_idx {sample_idx} is out of range. Max index: {len(filtered_errors) - 1}")
        sample_idx = 0
    
    # Get the original index in df_trj from the 'index' column (NOT the DataFrame row position)
    original_idx = filtered_errors.iloc[sample_idx]["index"]
    
    # Get error output and pred_solution
    eval_error_output = filtered_errors.iloc[sample_idx]["eval_error_output"]
    pred_solution = df_trj.iloc[original_idx]["rollouts"][0]["traj"]["transitions"][-1]["info"].get("pred_solution", None)
    task_name = filtered_errors.iloc[sample_idx].get("task_name", "Unknown")
    
    print(f"Error Type: {error_type}")
    print(f"Total samples with this error: {len(filtered_errors)}")
    print(f"Investigating sample {sample_idx} (original index: {original_idx})")
    print(f"Task: {task_name}")
    print("-" * 50)
    
    # if show_full_output:
    #     print("\n=== Eval Error Output ===")
    #     print(eval_error_output)
    #     print("\n=== Predicted Solution ===")
    #     print(pred_solution)
    
    return {
        'filtered_errors': filtered_errors,
        'error_counts': error_counts,
        'eval_error_output': eval_error_output,
        'pred_solution': pred_solution,
        'task_name': task_name,
        'original_idx': original_idx
    }
def filter_out_infra_errors(df_trj, df_trj_path=None, all_metrics_path=None):
    """
    Filter out rows with infrastructure errors from df_trj and optionally from all_metrics.jsonl.
    
    Infra errors are identified by checking if the last transition's info contains:
    - "agentboxbackend" AND "has no attribute" AND "container"
    
    Args:
        df_trj: DataFrame with trajectory data
        df_trj_path: Optional path to the original jsonl file. If provided, saves cleaned df to same folder with _cleaned suffix
        all_metrics_path: Optional path to all_metrics.jsonl file. If provided, removes corresponding rows and saves cleaned version.
                          Can also be inferred from df_trj_path if it follows the expected folder structure.
    
    Returns:
        df_clean: DataFrame with infra error rows removed
        df_infra_errors: DataFrame containing only the infra error rows
    """
    import os
    
    infra_error_indices = []
    
    for idx in range(len(df_trj)):
        try:
            row = df_trj.iloc[idx]
            rollouts = row.get("rollouts", [])
            if rollouts and len(rollouts) > 0:
                traj = rollouts[0].get("traj", {})
                transitions = traj.get("transitions", [])
                if transitions and len(transitions) > 0:
                    info = transitions[-1].get("info", {})
                    info_str = str(info).lower()
                    
                    if ("agentboxbackend" in info_str and 
                        "has no attribute" in info_str and 
                        "container" in info_str):
                        infra_error_indices.append(idx)
        except Exception as e:
            # Skip rows that can't be processed
            continue
    
    # Create mask for non-infra-error rows
    mask = ~df_trj.index.isin(df_trj.iloc[infra_error_indices].index)
    
    df_clean = df_trj[mask].copy()
    df_infra_errors = df_trj.iloc[infra_error_indices].copy()
    
    print(f"Total rows: {len(df_trj)}")
    print(f"Infra error rows removed: {len(df_infra_errors)}")
    print(f"Clean rows remaining: {len(df_clean)}")
    
    # Save cleaned df if path is provided
    if df_trj_path is not None:
        # Rename original file with _error suffix, then save cleaned df to original path
        if df_trj_path.endswith('.jsonl'):
            error_path = df_trj_path[:-6] + '_error.jsonl'
        else:
            error_path = df_trj_path + '_error'
        
        # Rename original file to _error
        if os.path.exists(df_trj_path):
            os.rename(df_trj_path, error_path)
            print(f"Original file renamed to: {error_path}")
        
        # Save cleaned df to original path
        df_clean.to_json(df_trj_path, orient='records', lines=True)
        print(f"Cleaned DataFrame saved to: {df_trj_path}")
        
        # Try to infer all_metrics_path if not provided
        if all_metrics_path is None:
            # df_trj_path is in .../trajectories/<subfolder>/<file>.jsonl
            # all_metrics.jsonl is in .../trajectories/
            trj_dir = os.path.dirname(df_trj_path)
            parent_dir = os.path.dirname(trj_dir)
            inferred_metrics_path = os.path.join(parent_dir, "all_metrics.jsonl")
            if os.path.exists(inferred_metrics_path):
                all_metrics_path = inferred_metrics_path
                print(f"Inferred all_metrics.jsonl path: {all_metrics_path}")
    
    # Clean all_metrics.jsonl if path is provided or inferred
    if all_metrics_path is not None and os.path.exists(all_metrics_path):
        try:
            df_metrics = pd.read_json(all_metrics_path, lines=True)
            original_metrics_count = len(df_metrics)
            
            # Remove the same indices from all_metrics
            metrics_mask = ~df_metrics.index.isin(df_metrics.iloc[infra_error_indices].index)
            df_metrics_clean = df_metrics[metrics_mask].copy()
            
            # Rename original all_metrics file with _error suffix
            if all_metrics_path.endswith('.jsonl'):
                error_metrics_path = all_metrics_path[:-6] + '_error.jsonl'
            else:
                error_metrics_path = all_metrics_path + '_error'
            
            os.rename(all_metrics_path, error_metrics_path)
            print(f"\nall_metrics.jsonl:")
            print(f"  Original file renamed to: {error_metrics_path}")
            
            # Save cleaned all_metrics to original path
            df_metrics_clean.to_json(all_metrics_path, orient='records', lines=True)
            print(f"  Total rows: {original_metrics_count}")
            print(f"  Rows removed: {original_metrics_count - len(df_metrics_clean)}")
            print(f"  Clean rows remaining: {len(df_metrics_clean)}")
            print(f"  Cleaned file saved to: {all_metrics_path}")
        except Exception as e:
            print(f"Warning: Could not process all_metrics.jsonl: {e}")
    elif all_metrics_path is not None:
        print(f"Warning: all_metrics.jsonl not found at: {all_metrics_path}")
    
    return df_clean, df_infra_errors