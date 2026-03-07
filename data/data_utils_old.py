import pandas as pd
import os
import gzip
import json
import zlib
import numpy as np
import sys
import re

import ast


import numpy as np
import importlib

# Pattern to match special LLM tokens like <|eot_id|>, <|start_header_id|>, etc.
SPECIAL_TOKEN_PATTERN = re.compile(r'<\|.*?\|>')

# Pattern to collapse multiple newlines into a single newline
MULTI_NEWLINE_PATTERN = re.compile(r'\n{2,}')

def build_rollout(df_trj, rollout_id: int) -> list[dict]:
    """
    Build a rollout as a list of turn dicts for a given rollout_id.
    
    Args:
        df_trj: DataFrame containing trajectory data
        rollout_id: Index of the rollout in the DataFrame
        
    Returns: 
        List of dicts, each with "action" and "observation" fields
    """
    transitions = df_trj.iloc[rollout_id]["rollouts"][0]["traj"]["transitions"]
    
    rollout = []
    for t, turn in enumerate(transitions):
        rollout.append({
            "turn_id": t,
            "action": turn["action_str"],
            "observation": turn["observation_str"]
        })
    
    return rollout

def build_rollout_str(df_trj, rollout_id: int) -> str:
    """
    Build a rollout as a concatenated string for a given rollout_id.
    
    Args:
        df_trj: DataFrame containing trajectory data
        rollout_id: Index of the rollout in the DataFrame
        
    Returns: 
        A string with all turns concatenated, each with turn_id, action, and observation.
    """
    transitions = df_trj.iloc[rollout_id]["rollouts"][0]["traj"]["transitions"]
    
    rollout_parts = []
    for t, turn in enumerate(transitions):
        step = t + 1  # 1-indexed steps
        # Clean special tokens from observation and normalize whitespace
        observation = SPECIAL_TOKEN_PATTERN.sub('', turn['observation_str'])
        observation = MULTI_NEWLINE_PATTERN.sub('\n', observation).strip()
        rollout_parts.append(
            f"Step {step} - Agent Action:\n{turn['action_str']}\n\n"
            f"Step {step} - Environment Response:\n{observation}\nassistant"
        )
    
    return "\n\n".join(rollout_parts)

def calculate_rollout_tokens(df_trj, rollout_id: int) -> tuple[int, int]:
    """
    Calculate total tokens in a rollout.
    
    Args:
        df_trj: DataFrame containing trajectory data
        rollout_id: Index of the rollout in the DataFrame
        
    Returns: 
        Tuple of (total_rollout_tokens, total_action_tokens)
        - total_rollout_tokens: sum of len(action) + len(observation) for all turns
        - total_action_tokens: sum of len(action) for all turns
    """
    transitions = df_trj.iloc[rollout_id]["rollouts"][0]["traj"]["transitions"]
    
    total_rollout_tokens = 0
    total_action_tokens = 0
    
    for turn in transitions:
        action_tokens = len(turn["action"])
        observation_tokens = len(turn["observation"])
        total_action_tokens += action_tokens
        total_rollout_tokens += action_tokens + observation_tokens
    
    return total_rollout_tokens, total_action_tokens

def flatten_dataframe(df_trj, only_valid_submissions: bool = False) -> pd.DataFrame:
    """
    Flatten the trajectory DataFrame into a simplified pandas DataFrame.
    
    Args:
        df_trj: DataFrame containing trajectory data
        only_valid_submissions: If True, only keep rollouts with valid_submission=True.
            Defaults to False.
    
    Returns:
        DataFrame with task_name, task_description, code, percentile, valid_submission,
        eval_error_output, eval_duration, rollout_duration, rollout, rollout_str,
        total_rollout_tokens, and total_action_tokens columns.
    """
    rows = []
    
    for idx in range(len(df_trj)):
        row = df_trj.iloc[idx]
        rollout_data = row["rollouts"][0]
        transitions = rollout_data["traj"]["transitions"]
        last_outcomes = transitions[-1]["outcomes"]
        info = transitions[-1]["info"]
        
        valid_submission = last_outcomes.get("valid_submission")
        
        if only_valid_submissions and not valid_submission:
            continue
        
        # total_rollout_tokens, total_action_tokens = calculate_rollout_tokens(df_trj, idx)
        
        rows.append({
            "task_name": rollout_data["start_args"]["instance_id"],
            "task_description": rollout_data["start_args"]["task_description"],
            "code": info.get("pred_solution"),
            "percentile": last_outcomes.get("percentile"),
            "valid_submission": valid_submission,
            "eval_error_output": last_outcomes.get("eval_error_output"),
            "eval_duration": last_outcomes.get("gpu_execution_duration"),
            "rollout_duration": rollout_data["metrics"].get("rollout/duration"),
            "rollout": build_rollout(df_trj, idx),
            "rollout_str": build_rollout_str(df_trj, idx),
            # "total_rollout_tokens": total_rollout_tokens,
            # "total_action_tokens": total_action_tokens
        })
    
    return pd.DataFrame(rows)



