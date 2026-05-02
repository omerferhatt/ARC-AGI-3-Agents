from typing import Type, cast

from dotenv import load_dotenv

from .agent import Agent, Playback
from .recorder import Recorder
from .swarm import Swarm
from .templates.langgraph_functional_agent import (
    LangGraphFunc,
    LangGraphTextOnly,
    OllamaDeepSeekLangGraphFunc,
    OllamaDeepSeekLangGraphTextOnly,
)
from .templates.langgraph_random_agent import LangGraphRandom
from .templates.langgraph_thinking import LangGraphThinking
from .templates.llm_agents import (
    LLM,
    FastLLM,
    GuidedLLM,
    OllamaDeepSeekFastLLM,
    OllamaDeepSeekLLM,
    ReasoningLLM,
)
from .templates.multimodal import MultiModalLLM
from .templates.random_agent import Random
from .templates.reasoning_agent import ReasoningAgent
from .templates.smolagents import (
    OllamaDeepSeekSmolCodingAgent,
    OllamaDeepSeekSmolVisionAgent,
    SmolCodingAgent,
    SmolVisionAgent,
)

load_dotenv()

def _agent_subclasses(cls: Type[Agent]) -> set[Type[Agent]]:
    subclasses: set[Type[Agent]] = set()
    for subclass in cls.__subclasses__():
        subclasses.add(cast(Type[Agent], subclass))
        subclasses.update(_agent_subclasses(cast(Type[Agent], subclass)))
    return subclasses


AVAILABLE_AGENTS: dict[str, Type[Agent]] = {
    cls.__name__.lower(): cls
    for cls in _agent_subclasses(Agent)
    if cls.__name__ != "Playback"
}

# add all the recording files as valid agent names
for rec in Recorder.list():
    AVAILABLE_AGENTS[rec] = Playback

# update the agent dictionary to include subclasses of LLM class
AVAILABLE_AGENTS["reasoningagent"] = ReasoningAgent

__all__ = [
    "Swarm",
    "Random",
    "LangGraphFunc",
    "LangGraphTextOnly",
    "LangGraphThinking",
    "LangGraphRandom",
    "LLM",
    "FastLLM",
    "ReasoningLLM",
    "GuidedLLM",
    "OllamaDeepSeekLLM",
    "OllamaDeepSeekFastLLM",
    "OllamaDeepSeekLangGraphFunc",
    "OllamaDeepSeekLangGraphTextOnly",
    "ReasoningAgent",
    "SmolCodingAgent",
    "SmolVisionAgent",
    "OllamaDeepSeekSmolCodingAgent",
    "OllamaDeepSeekSmolVisionAgent",
    "Agent",
    "Recorder",
    "Playback",
    "AVAILABLE_AGENTS",
    "MultiModalLLM",
]
