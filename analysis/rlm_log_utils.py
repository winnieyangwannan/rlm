import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


def load_rlm_log(log_path: str | Path) -> list[dict]:
    """
    Load an RLM log file and return list of parsed entries.
    
    The log file is JSONL format where each line is a JSON object.
    First line is metadata, subsequent lines are iterations.
    
    Returns:
        List of dicts - entries[0] is metadata, entries[1:] are iterations
    """
    entries = []
    with open(log_path, "r") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def get_final_answer(iterations: list[dict]) -> str | None:
    """
    Extract the final answer from iterations.
    
    Args:
        iterations: List of iteration dicts (entries[1:] from load_rlm_log)
        
    Returns:
        The final answer string, or None if not found
    """
    for it in reversed(iterations):
        if it.get("final_answer"):
            return it["final_answer"]
    return None


def get_total_runtime(entries: list[dict]) -> timedelta:
    """
    Calculate total runtime from log entries.
    
    Args:
        entries: Full list of entries from load_rlm_log (metadata + iterations)
        
    Returns:
        timedelta representing total runtime
    """
    if len(entries) < 2:
        return timedelta(0)
    
    start_time = datetime.fromisoformat(entries[0]["timestamp"])
    end_time = datetime.fromisoformat(entries[-1]["timestamp"])
    return end_time - start_time


# def get_total_iteration_time(iterations: list[dict]) -> float:
#     """
#     Sum of all iteration times (actual execution time, excludes setup overhead).
    
#     This is more accurate than timestamp-based runtime as it uses the
#     internally tracked iteration_time from time.perf_counter().
    
#     Args:
#         iterations: List of iteration dicts (entries[1:] from load_rlm_log)
        
#     Returns:
#         Total iteration time in seconds
#     """
#     return sum(it.get("iteration_time", 0) for it in iterations)


def get_all_code_with_results(iterations: list[dict]) -> list[dict[str, Any]]:
    """
    Extract all code blocks with their execution results.
    
    Args:
        iterations: List of iteration dicts (entries[1:] from load_rlm_log)
        
    Returns:
        List of dicts with keys: iteration, code, stdout, stderr, execution_time
    """
    all_code = []
    for it in iterations:
        iter_num = it.get("iteration", 0)
        for block in it.get("code_blocks", []):
            result = block.get("result", {})
            all_code.append({
                "iteration": iter_num,
                "code": block.get("code", ""),
                "stdout": result.get("stdout", ""),
                "stderr": result.get("stderr", ""),
                "execution_time": result.get("execution_time"),
            })
    return all_code




def get_all_code(iterations: list[dict]) -> list[str]:
    """
    Extract all code blocks executed during the RLM session.
    
    Args:
        iterations: List of iteration dicts (entries[1:] from load_rlm_log)
        
    Returns:
        List of code strings
    """
    all_code = []
    for it in iterations:
        for block in it.get("code_blocks", []):
            all_code.append(block["code"])
    return all_code


def get_sub_rlm_calls(iterations: list[dict]) -> list[dict[str, Any]]:
    """
    Extract all sub-LLM calls (llm_query / llm_query_batched) from iterations.
    
    Args:
        iterations: List of iteration dicts (entries[1:] from load_rlm_log)
        
    Returns:
        List of RLM call dicts with keys:
        - iteration, code_block_idx, root_model, prompt, response, 
        - execution_time, usage_summary
    """
    all_calls = []
    for it in iterations:
        iter_num = it.get("iteration", 0)
        for block_idx, block in enumerate(it.get("code_blocks", [])):
            for call in block.get("result", {}).get("rlm_calls", []):
                all_calls.append({
                    "iteration": iter_num,
                    "code_block_idx": block_idx,
                    "root_model": call.get("root_model"),
                    "prompt": call.get("prompt"),
                    "response": call.get("response"),
                    "execution_time": call.get("execution_time"),
                    "usage_summary": call.get("usage_summary"),
                })
    return all_calls


def get_sub_rlm_calls_summary(iterations: list[dict]) -> dict[str, Any]:
    """
    Get a summary of all RLM calls.
    
    Args:
        iterations: List of iteration dicts (entries[1:] from load_rlm_log)
        
    Returns:
        Dict with: total_calls, total_input_tokens, total_output_tokens,
                   calls_by_iteration, models_used
    """
    calls = get_sub_rlm_calls(iterations)
    
    total_input = 0
    total_output = 0
    calls_by_iter = {}
    models = set()
    
    for call in calls:
        iter_num = call["iteration"]
        calls_by_iter[iter_num] = calls_by_iter.get(iter_num, 0) + 1
        
        if call.get("root_model"):
            models.add(call["root_model"])
        
        usage = call.get("usage_summary", {})
        if usage:
            for model_usage in usage.get("model_usage_summaries", {}).values():
                total_input += model_usage.get("total_input_tokens", 0)
                total_output += model_usage.get("total_output_tokens", 0)
    
    return {
        "total_calls": len(calls),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "calls_by_iteration": calls_by_iter,
        "models_used": list(models),
    }


def extract_all(log_path: str | Path) -> dict[str, Any]:
    """
    Extract all key information from an RLM log in one call.
    
    Returns:
        Dict with: metadata, iterations, final_answer, code_blocks, 
                   rlm_calls, rlm_calls_summary, num_iterations
    """
    entries = load_rlm_log(log_path)
    metadata = entries[0] if entries else None
    iterations = entries[1:]
    
    return {
        "metadata": metadata,
        "iterations": iterations,
        "final_answer": get_final_answer(iterations),
        "code_blocks": get_all_code_with_results(iterations),
        "rlm_calls": get_sub_rlm_calls(iterations),
        "rlm_calls_summary": get_sub_rlm_calls_summary(iterations),
        "num_iterations": len(iterations),
    }


def archive_file(file_path: str, archive_dir: str = "/checkpoint/maui_sft/winnieyangwn/rlm_dumps/archive/gpt5/comparison/") -> list[str]:
    """Move a file and its corresponding .md file to the archive directory.
    
    Args:
        file_path: Path to the .jsonl file to archive
        archive_dir: Destination archive directory
        
    Returns:
        List of new paths of the archived files
    """
    import os
    archived_paths = []
    
    # Archive the main file
    dest_path = os.path.join(archive_dir, os.path.basename(file_path))
    if os.path.exists(dest_path):
        print(f"Skipped (already exists): {dest_path}")
    else:
        dest_path = shutil.move(file_path, archive_dir)
        print(f"Archived: {file_path} -> {dest_path}")
        archived_paths.append(dest_path)
    
    # Also archive the corresponding .md file if it exists
    if file_path.endswith(".jsonl"):
        md_path = file_path.replace(".jsonl", ".md")
        if os.path.exists(md_path):
            md_dest_path = os.path.join(archive_dir, os.path.basename(md_path))
            if os.path.exists(md_dest_path):
                print(f"Skipped (already exists): {md_dest_path}")
            else:
                md_dest_path = shutil.move(md_path, archive_dir)
                print(f"Archived: {md_path} -> {md_dest_path}")
                archived_paths.append(md_dest_path)
    
    return archived_paths