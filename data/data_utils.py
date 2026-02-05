import pandas as pd
import os
import gzip
import json
import zlib
import numpy as np
import sys


import ast


import numpy as np
import importlib

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

def flatten_dataframe(df_trj) -> pd.DataFrame:
    """
    Flatten the trajectory DataFrame into a simplified pandas DataFrame.
    
    Returns:
        DataFrame with task_name, task_description, code, percentile, valid_submission,
        eval_error_output, eval_duration, rollout_duration, and rollout columns.
    """
    rows = []
    
    for idx in range(len(df_trj)):
        row = df_trj.iloc[idx]
        rollout_data = row["rollouts"][0]
        transitions = rollout_data["traj"]["transitions"]
        last_outcomes = transitions[-1]["outcomes"]
        info = transitions[-1]["info"]
        
        rows.append({
            "task_name": rollout_data["start_args"]["instance_id"],
            "task_description": rollout_data["start_args"]["task_description"],
            "code": info.get("pred_solution"),
            "percentile": last_outcomes.get("percentile"),
            "valid_submission": last_outcomes.get("valid_submission"),
            "eval_error_output": last_outcomes.get("eval_error_output"),
            "eval_duration": last_outcomes.get("gpu_execution_duration"),
            "rollout_duration": rollout_data["metrics"].get("rollout/duration"),
            "rollout": build_rollout(df_trj, idx)
        })
    
    return pd.DataFrame(rows)



