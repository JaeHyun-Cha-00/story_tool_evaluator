"""Configuration for the MCP story evaluation server."""

from dataclasses import dataclass

@dataclass(frozen=True)
class WolverineSettings:
    """Runtime configuration for the Wolverine OpenAI-compatible endpoint."""

    base_url: str = "http://localhost:8000/v1"  # vLLM Server Address
    api_key: str = "EMPTY"
    model: str = "Qwen/Qwen3-4B-Instruct-2507"  # Model Name
    # meta-llama/Llama-3.1-8B-Instruct
    # meta-llama/Llama-3.2-3B-Instruct
    # google/gemma-3-4b-it
    # google/gemma-3-12b-it
    # Qwen/Qwen3-4B-Instruct-2507
    # Qwen/Qwen2.5-7B-Instruct

    temperature: float = 0.7
    min_p: float = 0.1
    # max_tokens: int = 1500 

WOLVERINE_SETTINGS = WolverineSettings()