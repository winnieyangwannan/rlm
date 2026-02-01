# RLM Output Guide

## 3. Log File Structure (JSONL)

RLM logs are stored as JSON Lines (`.jsonl`) files where each line is a complete JSON object.

### File Location
```
./logs/rlm_{timestamp}_{uuid}.jsonl
```

### Entry Types

#### 1. Metadata (First Line)
```json
{
  "type": "metadata",
  "timestamp": "2026-01-31T11:05:04.339193",
  "root_model": "gpt-5",
  "max_depth": 1,
  "max_iterations": 10,
  "backend": "azure_openai",
  "backend_kwargs": {
    "model_name": "gpt-5",
    "azure_endpoint": "https://...",
    "azure_deployment": "gpt-5",
    "api_version": "2025-03-01-preview"
  },
  "environment_type": "local",
  "environment_kwargs": {},
  "other_backends": null
}
```

#### 2. Iteration Entries (Subsequent Lines)
```json
{
  "type": "iteration",
  "iteration": 1,
  "timestamp": "2026-01-31T11:05:16.345724",
  "prompt": [...],
  "response": "...",
  "code_blocks": [...],
  "final_answer": null,
  "iteration_time": 12.5
}
```

### Iteration Entry Fields

| Field | Type | Description |
|-------|------|-------------|
| `type` | `"iteration"` | Entry type identifier |
| `iteration` | `int` | 1-indexed iteration number |
| `timestamp` | `str` | ISO format timestamp |
| `prompt` | `str \| list[dict]` | The prompt sent to the LLM |
| `response` | `str` | Raw LLM response text |
| `code_blocks` | `list[CodeBlock]` | Executed code and results |
| `final_answer` | `str \| null` | Set when `FINAL_VAR()` called |
| `iteration_time` | `float` | Seconds for this iteration |

### Code Block Structure
```json
{
  "code": "print(len(context))",
  "result": {
    "stdout": "4800\n",
    "stderr": "",
    "locals": {"context": "<list of 4800 items>"}, #  captures al user-defined variables in the REPL environment after the code executes.
    "execution_time": 0.001,
    "rlm_calls": [...]
  }
}
```

### RLM Call Structure (Sub-LLM Calls)
When code uses `llm_query()` or `llm_query_batched()`:
```json
{
  "root_model": "gpt-5",
  "prompt": "Analyze this data...",
  "response": "The analysis shows...",
  "execution_time": 2.5,
  "usage_summary": {
    "model_usage_summaries": {
      "gpt-5": {
        "total_calls": 1,
        "total_input_tokens": 1500,
        "total_output_tokens": 500
      }
    }
  }
}
```

### Schema Tree
```
Log File (.jsonl)
├── Line 1: Metadata
│   ├── type: "metadata"
│   ├── timestamp: str
│   ├── root_model: str
│   ├── max_depth: int
│   ├── max_iterations: int
│   ├── backend: str
│   ├── backend_kwargs: dict
│   ├── environment_type: str
│   ├── environment_kwargs: dict
│   └── other_backends: list | null
│
└── Lines 2+: Iterations
    ├── type: "iteration"
    ├── iteration: int
    ├── timestamp: str
    ├── prompt: str | list[dict]
    ├── response: str
    ├── code_blocks: list
    │   └── [CodeBlock]
    │       ├── code: str
    │       └── result: REPLResult
    │           ├── stdout: str
    │           ├── stderr: str
    │           ├── locals: dict
    │           ├── execution_time: float
    │           └── rlm_calls: list
    │               └── [RLMChatCompletion]
    │                   ├── root_model: str
    │                   ├── prompt: str | dict
    │                   ├── response: str
    │                   ├── execution_time: float
    │                   └── usage_summary: UsageSummary
    ├── final_answer: str | null
    └── iteration_time: float
```

### Parsing Example
```python
import json
from pathlib import Path

def load_rlm_log(log_path: str) -> list[dict]:
    """Load all entries from an RLM log file."""
    entries = []
    with open(log_path, "r") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries

# Usage
entries = load_rlm_log("./logs/rlm_2026-01-31_11-05-04_ee149f54.jsonl")
metadata = entries[0]  # First entry is metadata
iterations = entries[1:]  # Rest are iterations

# Get final answer
for it in reversed(iterations):
    if it.get("final_answer"):
        print(it["final_answer"])
        break

# Get all code
for it in iterations:
    for block in it.get("code_blocks", []):
        print(block["code"])

# Get all sub-LLM calls
for it in iterations:
    for block in it.get("code_blocks", []):
        for call in block.get("result", {}).get("rlm_calls", []):
            print(f"Prompt: {call['prompt'][:100]}...")
            print(f"Response: {call['response'][:100]}...")
```
