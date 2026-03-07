#!/bin/bash

# Example launch script for system_consolidation.py
# run_id, model, and group_name are extracted from episodic_memory_paths automatically
# Now supports multiple episodic memory paths - the best rollout is selected across ALL sources
# Use --do_not_skip to force re-processing specific tasks even if output already exists
#   e.g., --do_not_skip "task1" "task2"



##################################################################################
# 531 --> 531
##################################################################################


# GENERATION 0
# Example with multiple paths (best rollout selected across all sources):
# python /home/winnieyangwn/rlm/experiments/memory/memory_system_consolidation/system_consolidation.py \
#     --episodic_memory_paths \
#         "/checkpoint/agentic-models/winnieyangwn/memory/gpt5/531/531/episodic_memory/gen0/episodic_memory_gen0.jsonl" \
#     --semantic_memory_dir \
#         "/checkpoint/agentic-models/winnieyangwn/memory/gpt5/531/531/semantic_memory/gen0/" \
#     --consolidation_mode "best_code" \
#     --max_workers 4 \
#     --do_not_skip "kuzushiji-recognition" # example tasks to re-process even if output already exists


# # GENERATION 1
# # Example with multiple paths (best rollout selected across all sources):
# python /home/winnieyangwn/rlm/experiments/memory/memory_system_consolidation/system_consolidation.py \
#     --episodic_memory_paths \
#         "/checkpoint/agentic-models/winnieyangwn/memory/gpt5/531/531/episodic_memory/gen0/episodic_memory_gen0.jsonl" \
#         "/checkpoint/agentic-models/winnieyangwn/memory/gpt5/531/531/episodic_memory/gen0/episodic_memory_gen1.jsonl" \
#     --semantic_memory_dir \
#         "/checkpoint/agentic-models/winnieyangwn/memory/gpt5/531/531/semantic_memory/gen1/" \
#     --consolidation_mode "best_code" \
#     --max_workers 4 \
#     --do_not_skip "kuzushiji-recognition" # example tasks to re-process even if output already exists

##################################################################################
# 531 --> 532
##################################################################################


# GENERATION 0
# Example with multiple paths (best rollout selected across all sources):
python /home/winnieyangwn/rlm/experiments/memory/memory_system_consolidation/system_consolidation.py \
    --episodic_memory_paths \
        "/checkpoint/agentic-models/winnieyangwn/memory/gpt5/531/532/episodic_memory/gen0/episodic_memory_gen0.jsonl" \
    --semantic_memory_dir \
        "/checkpoint/agentic-models/winnieyangwn/memory/gpt5/531/532/semantic_memory/gen0/" \
    --consolidation_mode "best_code" \
    --max_workers 4 \
