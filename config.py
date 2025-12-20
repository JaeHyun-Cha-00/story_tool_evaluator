"""Configuration for the MCP story evaluation server."""

from dataclasses import dataclass

@dataclass(frozen=True)
class WolverineSettings:
    """Runtime configuration for the Wolverine OpenAI-compatible endpoint."""

    base_url: str = "http://localhost:8000/v1"  # vLLM Server Address
    api_key: str = "EMPTY"
    model: str = "meta-llama/Llama-3.1-8B-Instruct"  # Model Name

# Single shared settings instance used across the application.
WOLVERINE_SETTINGS = WolverineSettings()