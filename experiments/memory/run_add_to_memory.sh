#!/bin/bash

# Example launch script for add_to_memory.py
# Converts raw trajectories into episodic memories (flattened metadata)



##################################################################################
# 531 --> 531
##################################################################################


# # GENERATION 0
# python /home/winnieyangwn/rlm/experiments/memory/add_to_memory.py \
#     --run_id "531_mle_30_7" \
#     --group "agentic-models" \
#     --memory_path "/checkpoint/agentic-models/winnieyangwn/memory/gpt5/531/531/episodic_memory/" \
#     --generation_id "gen0" \
#     --only_valid_submissions false

# python /home/winnieyangwn/rlm/experiments/memory/add_to_memory.py \
#     --run_id "531_mle_30_23" \
#     --group "agentic-models" \
#     --memory_path "/checkpoint/agentic-models/winnieyangwn/memory/gpt5/531/531/episodic_memory/" \
#     --generation_id "gen0" \
#     --only_valid_submissions false



# # GENERATION 1
# python /home/winnieyangwn/rlm/experiments/memory/add_to_memory.py \
#     --run_id "531_mle_30_7_r1" \
#     --group "agentic-models" \
#     --memory_path "/checkpoint/agentic-models/winnieyangwn/memory/gpt5/531/531/episodic_memory/" \
#     --generation_id "gen1" \
#     --only_valid_submissions false



##################################################################################
# 531 --> 532
##################################################################################


# GENERATION 0
# python /home/winnieyangwn/rlm/experiments/memory/add_to_memory.py \
#     --run_id "531_mle_30_7" \
#     --group "agentic-models" \
#     --memory_path "/checkpoint/agentic-models/winnieyangwn/memory/gpt5/531/532/episodic_memory/" \
#     --generation_id "gen0" \
#     --only_valid_submissions false

python /home/winnieyangwn/rlm/experiments/memory/add_to_memory.py \
    --run_id "531_mle_30_23" \
    --group "agentic-models" \
    --memory_path "/checkpoint/agentic-models/winnieyangwn/memory/gpt5/531/532/episodic_memory/" \
    --generation_id "gen0" \
    --only_valid_submissions false



# # GENERATION 1
# python /home/winnieyangwn/rlm/experiments/memory/add_to_memory.py \
#     --run_id "532_mle_30_7_r1" \
#     --group "agentic-models" \
#     --memory_path "/checkpoint/agentic-models/winnieyangwn/memory/gpt5/531/532/episodic_memory/" \
#     --generation_id "gen1" \
#     --only_valid_submissions false
