"""Helpers for OpenAI-compatible model endpoints."""

import os
from typing import Any

from openai import OpenAI

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"


def _with_v1_path(url: str) -> str:
    """Return an OpenAI-compatible base URL for providers exposing /v1."""
    cleaned = url.rstrip("/")
    if cleaned.endswith("/v1"):
        return cleaned
    return f"{cleaned}/v1"


def get_openai_base_url() -> str | None:
    """Read the configured OpenAI-compatible endpoint, including Ollama."""
    if base_url := os.environ.get("OPENAI_BASE_URL"):
        return base_url.rstrip("/")

    if ollama_base_url := os.environ.get("OLLAMA_BASE_URL"):
        return _with_v1_path(ollama_base_url)

    if ollama_host := os.environ.get("OLLAMA_HOST"):
        return _with_v1_path(ollama_host)

    if os.environ.get("OPENAI_PROVIDER", "").lower() == "ollama":
        return DEFAULT_OLLAMA_BASE_URL

    return None


def get_openai_api_key(api_key_env: str = "OPENAI_API_KEY") -> str:
    """Return an API key, using Ollama's accepted placeholder when needed."""
    if api_key := os.environ.get(api_key_env):
        return api_key

    if api_key_env != "OPENAI_API_KEY" and (api_key := os.environ.get("OPENAI_API_KEY")):
        return api_key

    if get_openai_base_url():
        return "ollama"

    return ""


def create_openai_client(api_key_env: str = "OPENAI_API_KEY") -> OpenAI:
    """Create an OpenAI client that can target OpenAI-compatible local servers."""
    kwargs: dict[str, Any] = {"api_key": get_openai_api_key(api_key_env)}
    if base_url := get_openai_base_url():
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def chat_openai_kwargs() -> dict[str, Any]:
    """Return kwargs for langchain_openai.ChatOpenAI."""
    kwargs: dict[str, Any] = {"api_key": get_openai_api_key()}
    if base_url := get_openai_base_url():
        kwargs["base_url"] = base_url
    return kwargs
