"""Utilities for inspecting RLM log files."""

import json

# ANSI color codes
RESET = "\033[0m"
CYAN = "\033[36m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"


def load_log(log_path: str) -> list:
    """Load a JSONL log file.
    
    Args:
        log_path: Path to the JSONL log file
        
    Returns:
        List of log entries
    """
    with open(log_path, 'r') as f:
        log_data = [json.loads(line) for line in f]
    print(f"Loaded {len(log_data)} log entries")
    return log_data


def print_num_iterations(log_data: list) -> None:
    """Print the number of iterations in the log data.
    
    Args:
        log_data: List of log entries loaded from JSONL
    """
    # First entry is metadata, rest are iterations
    num_iterations = len(log_data) - 1
    print(f"Number of iterations: {num_iterations}")


def print_prompt(log_data: list, iteration: int) -> None:
    """Print the prompt for a given iteration.
    
    Args:
        log_data: List of log entries loaded from JSONL
        iteration: Iteration number (1-indexed)
    """
    # Role to color mapping
    role_colors = {
        "system": YELLOW,
        "user": CYAN,
        "assistant": GREEN,
    }
    
    entry = log_data[iteration]
    prompt = entry.get("prompt", [])
    
    for i, message in enumerate(prompt):
        role = message.get("role", "unknown")
        content = message.get("content", "")
        color = role_colors.get(role, RESET)
        print(f"{'='*60}")
        print(f"{color}[{i}] Role: {role}{RESET}")
        print(f"{'='*60}")
        print(f"{color}{content}{RESET}")
        print()


def print_response(log_data: list, iteration: int) -> None:
    """Print the response for a given iteration.
    
    Args:
        log_data: List of log entries loaded from JSONL
        iteration: Iteration number (1-indexed)
    """
    entry = log_data[iteration]
    response = entry.get("response", "")
    print(f"{'='*60}")
    print(f"Iteration {iteration} Response")
    print(f"{'='*60}")
    print(response)


def print_code_blocks(log_data: list, iteration: int) -> None:
    """Print all code blocks for a given iteration.
    
    Args:
        log_data: List of log entries loaded from JSONL
        iteration: Iteration number (1-indexed)
    """
    entry = log_data[iteration]
    code_blocks = entry.get("code_blocks", [])
    
    print(f"Iteration {iteration}: {len(code_blocks)} code block(s)")
    print()
    
    for i, block in enumerate(code_blocks):
        print(f"{'='*60}")
        print(f"Code Block {i}")
        print(f"{'='*60}")
        print(f"{CYAN}{block.get('code', '')}{RESET}")
        print()
        
        result = block.get("result", {})
        stdout = result.get("stdout", "")
        stderr = result.get("stderr", "")
        
        if stdout:
            print(f"{'-'*60}")
            print(f"{GREEN}stdout:{RESET}")
            print(f"{'-'*60}")
            print(f"{GREEN}{stdout}{RESET}")
        
        if stderr:
            print(f"{'-'*60}")
            print(f"{RED}stderr:{RESET}")
            print(f"{'-'*60}")
            print(f"{RED}{stderr}{RESET}")
        print()


def print_final_answer(log_data: list) -> None:
    """Print the final answer from the log data.
    
    Args:
        log_data: List of log entries loaded from JSONL
    """
    # Final answer is typically in metadata (first entry) or last entry
    metadata = log_data[0]
    final_answer = metadata.get("final_answer", None)
    
    if final_answer is None:
        # Try last entry
        last_entry = log_data[-1]
        final_answer = last_entry.get("final_answer", None)
    
    print(f"{'='*60}")
    print("Final Answer")
    print(f"{'='*60}")
    if final_answer is not None:
        print(final_answer)
    else:
        print("No final answer found")
