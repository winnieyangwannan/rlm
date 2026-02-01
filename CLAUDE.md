# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Recursive Language Models (RLMs) is an inference paradigm that enables language models to handle near-infinite length contexts by programmatically examining, decomposing, and recursively calling itself over its input. RLMs replace standard `llm.completion(prompt, model)` calls with `rlm.completion(prompt, model)` that offload context to a REPL environment where the LM can interact and launch sub-LM calls.

## Core Architecture

### Three-Layer Design

1. **Core Layer** (`rlm/core/`):
   - `rlm.py`: Main RLM class orchestrating the recursive inference loop
   - `lm_handler.py`: Multi-threaded socket server routing LLM requests between RLM process and environment subprocesses (uses 4-byte length prefix + JSON payload protocol)
   - `types.py`: Type definitions including `RLMChatCompletion`, `CodeBlock`, `REPLResult`, usage tracking types
   - `comms_utils.py`: Socket communication utilities for serializing/deserializing requests between host and sandboxes
   - **IMPORTANT**: Avoid touching core files unless necessary (see CONTRIBUTING.md)

2. **Clients Layer** (`rlm/clients/`):
   - `base_lm.py`: Abstract base class for LLM clients
   - Implementations: `openai.py`, `anthropic.py`, `gemini.py`, `azure_openai.py`, `portkey.py`, `litellm.py`
   - `__init__.py`: `get_client()` factory function routes backend string to appropriate client
   - Special handling: `vllm`, `openrouter`, `vercel` use OpenAI client with custom `base_url`

3. **Environments Layer** (`rlm/environments/`):
   - `base_env.py`: Abstract base class defining REPL interface and `SupportsPersistence` protocol
   - `local_repl.py`: Default environment using Python `exec()` in host process
   - `docker_repl.py`: Isolated Docker container environment
   - `modal_repl.py`: Cloud-based Modal Sandboxes
   - `prime_repl.py`: Prime Intellect Sandboxes (beta, may have slow runtimes)
   - `daytona_repl.py`: Daytona sandbox support
   - `__init__.py`: `get_environment()` factory routes environment string to implementation

### Key Execution Flow

1. User calls `RLM.completion(prompt)` which spawns an environment and LM handler
2. RLM builds system prompt and sends initial message to LLM
3. LLM responds with code blocks in markdown fenced format
4. RLM extracts code blocks, executes them in the REPL environment
5. If code contains recursive `lm.completion()` calls, they're routed through the LM handler socket server
6. Results are formatted and fed back to the LLM for next iteration
7. Loop continues until final answer is extracted or max iterations reached
8. Environment and handler are cleaned up on completion

## Development Commands

```bash
# Installation
make install              # Install base dependencies with uv
make install-dev         # Install dev + test dependencies
make install-modal       # Install Modal sandbox support
uv pip install -e ".[prime]"  # Install Prime sandbox support

# Development
make lint                # Run ruff linter
make format              # Run ruff formatter
make test                # Run pytest tests
make check               # Run lint + format + tests

# Examples
make quickstart          # Run quickstart example (needs OPENAI_API_KEY)
make docker-repl         # Run Docker REPL example (needs Docker installed)
make modal-repl          # Run Modal REPL example (needs Modal setup)
```

## Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_local_repl.py

# Run with coverage
uv run pytest --cov=rlm

# Run specific test function
uv run pytest tests/test_parsing.py::test_find_code_blocks
```

## Common Patterns

### Adding a New Client Backend

1. Create new file in `rlm/clients/` (e.g., `my_provider.py`)
2. Extend `BaseLM` class with required methods: `chat_completion()`, `async_chat_completion()`
3. Add backend to `ClientBackend` Literal type in `rlm/core/types.py`
4. Register in `get_client()` factory in `rlm/clients/__init__.py`
5. Add tests in `tests/clients/`

### Adding a New Environment

1. Create new file in `rlm/environments/` (e.g., `my_sandbox.py`)
2. Extend `BaseEnv` class implementing `execute()` and `cleanup()` methods
3. Optionally implement `SupportsPersistence` protocol for multi-turn persistence
4. Add environment to `EnvironmentType` Literal in `rlm/core/types.py`
5. Register in `get_environment()` factory in `rlm/environments/__init__.py`
6. Add tests in `tests/repl/`

### Usage Tracking

The RLM tracks token usage across all LLM calls (including recursive sub-calls):
- `ModelUsageSummary`: Per-model usage (calls, input tokens, output tokens)
- `UsageSummary`: Aggregates all model usage summaries
- Access via `result.usage_summary` after completion
- Usage propagates from sub-calls to parent calls through socket protocol

### Logging and Visualization

```python
from rlm.logger import RLMLogger

logger = RLMLogger(log_dir="./logs")
rlm = RLM(..., logger=logger, verbose=True)
```

Logs are saved as `.jsonl` files viewable in the visualizer:
```bash
cd visualizer/
npm run dev  # Runs on localhost:3001
```

## Configuration Notes

### Environment Variables

Set API keys as environment variables or use `.env` file (loaded automatically):
- `OPENAI_API_KEY`: For OpenAI client
- `ANTHROPIC_API_KEY`: For Anthropic client
- `PORTKEY_API_KEY`: For Portkey router
- `PRIME_API_KEY`: For Prime sandboxes
- `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`: For Azure OpenAI

### RLM Parameters

Key parameters when initializing `RLM()`:
- `backend`: LLM provider (`"openai"`, `"anthropic"`, `"gemini"`, etc.)
- `backend_kwargs`: Dict passed to client (e.g., `{"model_name": "gpt-4"}`)
- `environment`: Execution environment (`"local"`, `"docker"`, `"modal"`, `"prime"`, `"daytona"`)
- `environment_kwargs`: Dict passed to environment
- `max_depth`: Maximum recursive call depth (default: 1)
- `max_iterations`: Maximum iterations per completion (default: 30)
- `custom_system_prompt`: Override default RLM system prompt
- `other_backends`: List of additional backends available for recursive calls

## Important Constraints

- The repository aims to stay minimal and readable - avoid over-engineering
- Core files (`rlm/core/`) should rarely be modified
- Security: `local` environment uses `exec()` and is NOT production-safe for untrusted prompts
- Docker, Modal, Prime, and Daytona environments provide isolation for production use
- Code blocks must use markdown fenced code blocks (triple backticks with `python` language tag)
- Final answers must be marked with `FINAL_ANSWER:` prefix
