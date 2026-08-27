"""Conversation memory: bounded, per-session, and trust-tagged.

Memory is an injection surface in its own right -- the "multi-turn setup" attack
plants an instruction in turn 1 and detonates it in turn 3. Two rules hold here:

* only `user` and `assistant` turns are stored; retrieved chunk text is never
  written into memory, so it cannot be replayed later as if it were dialogue;
* every buffer is bounded (turns and characters), so a long session cannot grow
  the prompt without limit.

`history()` returns prior *user* turns only -- that is what the 31M rewriter
needs to resolve "what about the second one?" into a standalone query.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, Field

from localmind.agent.state import Clock, WallClock

__all__ = ["ConversationMemory", "SessionStore", "Turn"]

Role = Literal["user", "assistant"]


class Turn(BaseModel):
    role: Role
    content: str
    at: float = 0.0
    refused: bool = False


class ConversationMemory:
    """A bounded ring of turns for one session."""

    def __init__(
        self,
        session_id: str = "default",
        *,
        max_turns: int = 20,
        max_chars: int = 6000,
        max_turn_chars: int = 2000,
        clock: Clock | None = None,
    ) -> None:
        self.session_id = session_id
        self.max_turns = max_turns
        self.max_chars = max_chars
        self.max_turn_chars = max_turn_chars
        self.clock: Clock = clock or WallClock()
        self.turns: list[Turn] = []

    def _append(self, role: Role, content: str, refused: bool = False) -> Turn:
        turn = Turn(
            role=role, content=content[: self.max_turn_chars], at=self.clock.now(), refused=refused
        )
        self.turns.append(turn)
        self._trim()
        return turn

    def _trim(self) -> None:
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns :]
        total = sum(len(t.content) for t in self.turns)
        while total > self.max_chars and len(self.turns) > 1:
            total -= len(self.turns[0].content)
            self.turns.pop(0)

    def add_user(self, content: str) -> Turn:
        return self._append("user", content)

    def add_assistant(self, content: str, *, refused: bool = False) -> Turn:
        return self._append("assistant", content, refused=refused)

    def history(self, limit: int = 5) -> list[str]:
        """Prior user turns, oldest first -- the input to `ControlPlane.rewrite`."""
        return [t.content for t in self.turns if t.role == "user"][-limit:]

    def render(self, limit: int = 6) -> str:
        return "\n".join(f"{t.role}: {t.content}" for t in self.turns[-limit:])

    def clear(self) -> None:
        self.turns.clear()

    def __len__(self) -> int:
        return len(self.turns)


class SessionStore:
    """LRU map of session id -> memory. Bounded so a fuzzer cannot exhaust RAM."""

    def __init__(self, max_sessions: int = 128, clock: Clock | None = None, **memory_kw: int):
        self.max_sessions = max_sessions
        self.clock: Clock = clock or WallClock()
        self.memory_kw = memory_kw
        self._sessions: OrderedDict[str, ConversationMemory] = OrderedDict()

    def get(self, session_id: str) -> ConversationMemory:
        mem = self._sessions.get(session_id)
        if mem is None:
            mem = ConversationMemory(session_id, clock=self.clock, **self.memory_kw)
            self._sessions[session_id] = mem
            while len(self._sessions) > self.max_sessions:
                self._sessions.popitem(last=False)
        else:
            self._sessions.move_to_end(session_id)
        return mem

    def drop(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def ids(self) -> Iterable[str]:
        return list(self._sessions)

    def __len__(self) -> int:
        return len(self._sessions)


class MemorySnapshot(BaseModel):
    """Serialisable view of a session, for observability and tests."""

    session_id: str
    turns: list[Turn] = Field(default_factory=list)

    @classmethod
    def of(cls, memory: ConversationMemory) -> MemorySnapshot:
        return cls(session_id=memory.session_id, turns=list(memory.turns))
