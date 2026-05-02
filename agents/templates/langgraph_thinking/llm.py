from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from agents.templates.openai_client import chat_openai_kwargs

from .schema import LLM


def get_llm(llm: LLM) -> BaseChatModel:
    """
    Get an LLM instance based on the LLM enum.
    """

    match llm:
        case LLM.OPENAI_GPT_41:
            return ChatOpenAI(model="gpt-4.1", **chat_openai_kwargs())
        case _:
            raise ValueError(f"Unknown LLM: {llm}")
