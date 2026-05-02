"""Compatibility models for older examples and tests.

Runtime code uses `arcengine` and `arc_agi.scorecard` directly. This module
keeps the previous `agents.structs` import path usable while mapping old field
names like `score` onto the current `levels_completed` vocabulary.
"""

from typing import Any

from arcengine import ActionInput, GameAction, GameState
from arcengine import FrameData as _FrameData
from pydantic import BaseModel, Field, model_validator


class FrameData(_FrameData):
    @model_validator(mode="before")
    @classmethod
    def _map_score(cls, data: Any) -> Any:
        if isinstance(data, dict) and "score" in data and "levels_completed" not in data:
            data = data.copy()
            data["levels_completed"] = data.pop("score")
        return data

    @property
    def score(self) -> int:
        return self.levels_completed

    @score.setter
    def score(self, value: int) -> None:
        self.levels_completed = value

    def is_empty(self) -> bool:
        return not self.frame


class Card(BaseModel):
    game_id: str
    total_plays: int = 0
    scores: list[int] = Field(default_factory=list)
    states: list[GameState] = Field(default_factory=list)
    actions: list[int] = Field(default_factory=list)
    resets: list[int] = Field(default_factory=list)

    @property
    def idx(self) -> int:
        return len(self.scores) - 1

    @property
    def started(self) -> bool:
        return self.total_plays > 0

    @property
    def score(self) -> int | None:
        return self.scores[-1] if self.scores else None

    @property
    def high_score(self) -> int:
        return max(self.scores, default=0)

    @property
    def state(self) -> GameState:
        return self.states[-1] if self.states else GameState.NOT_PLAYED

    @property
    def action_count(self) -> int:
        return self.actions[-1] if self.actions else 0

    @property
    def total_actions(self) -> int:
        return sum(self.actions)


class Scorecard(BaseModel):
    card_id: str = ""
    api_key: str | None = None
    cards: dict[str, Card] = Field(default_factory=dict)

    @property
    def won(self) -> int:
        return sum(1 for card in self.cards.values() if card.state is GameState.WIN)

    @property
    def played(self) -> int:
        return sum(1 for card in self.cards.values() if card.started)

    @property
    def total_actions(self) -> int:
        return sum(card.total_actions for card in self.cards.values())

    def get(self, game_id: str | None = None) -> dict[str, Any]:
        if game_id:
            card = self.cards.get(game_id)
            return {game_id: card.model_dump()} if card else {}
        return {gid: card.model_dump() for gid, card in self.cards.items()}

    def get_json_for(self, game_id: str) -> dict[str, Any]:
        return {
            "won": self.won,
            "played": self.played,
            "total_actions": self.total_actions,
            "cards": self.get(game_id),
        }


__all__ = [
    "ActionInput",
    "Card",
    "FrameData",
    "GameAction",
    "GameState",
    "Scorecard",
]
