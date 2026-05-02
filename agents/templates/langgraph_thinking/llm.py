import os
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from .schema import LLM

DEFAULT_OLLAMA_OPENAI_BASE_URL = "http://localhost:11434/v1"


def _with_v1_path(url: str) -> str:
    cleaned = url.rstrip("/")
    if cleaned.endswith("/v1"):
        return cleaned
    return f"{cleaned}/v1"


def _openai_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if api_key := os.environ.get("OPENAI_API_KEY"):
        kwargs["api_key"] = api_key
    if base_url := os.environ.get("OPENAI_BASE_URL"):
        kwargs["base_url"] = base_url.rstrip("/")
    return kwargs


def _ollama_kwargs() -> dict[str, Any]:
    base_url = os.environ.get("OPENAI_BASE_URL") or DEFAULT_OLLAMA_OPENAI_BASE_URL
    return {
        "api_key": os.environ.get("OPENAI_API_KEY", "ollama"),
        "base_url": _with_v1_path(base_url),
    }


def get_llm(llm: LLM) -> BaseChatModel:
    """
    Get an LLM instance based on the LLM enum.
    """

    match llm:
        case LLM.OPENAI_GPT_41:
            return ChatOpenAI(model="gpt-4.1", **_openai_kwargs())
        case LLM.OLLAMA_DEEPSEEK_V4_FLASH:
            return ChatOpenAI(model="deepseek-v4-flash", **_ollama_kwargs())
        case LLM.OLLAMA_LLAMA_3_2:
            return ChatOpenAI(model="llama3.2", **_ollama_kwargs())
        case _:
            raise ValueError(f"Unknown LLM: {llm}")
