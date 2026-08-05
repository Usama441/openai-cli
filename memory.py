from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class Message:
    role: str
    content: str


class ConversationMemory:
    def __init__(self, max_messages: int = 20):
        self.max_messages = max_messages
        self.messages: List[Message] = []

    def add_user(self, content: str) -> None:
        self.messages.append(Message(role="user", content=content.strip()))
        self._trim()

    def add_assistant(self, content: str) -> None:
        self.messages.append(Message(role="assistant", content=content.strip()))
        self._trim()

    def _trim(self) -> None:
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

    def recent(self, limit: int | None = None) -> List[Message]:
        if limit is None:
            return list(self.messages)
        return list(self.messages)[-limit:]

    def as_history_strings(self, limit: int | None = None) -> List[str]:
        return [f"{msg.role.title()}: {msg.content}" for msg in self.recent(limit)]
