from typing import Type, cast

from dotenv import load_dotenv

from .agent import Agent, Playback
from .recorder import Recorder
from .swarm import Swarm
from .templates.langgraph_thinking import LangGraphThinking

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

__all__ = [
    "Swarm",
    "LangGraphThinking",
    "Agent",
    "Recorder",
    "Playback",
    "AVAILABLE_AGENTS",
]
