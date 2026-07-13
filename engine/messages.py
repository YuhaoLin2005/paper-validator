"""Message dataclasses for agent conversation (imported by api_client, query, main).

Serializes to OpenAI-compatible chat completions wire format.
No data files — pure dataclass definitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SystemMessage:
    content: str
    role: str = "system"

    def to_api(self) -> dict:
        return {"role": "system", "content": self.content}


@dataclass
class UserMessage:
    content: str
    role: str = "user"

    def to_api(self) -> dict:
        return {"role": "user", "content": self.content}


@dataclass
class AssistantMessage:
    content: str = ""
    role: str = "assistant"
    tool_calls: list[dict] = field(default_factory=list)
    logprobs: Optional[dict] = None

    def to_api(self) -> dict:
        msg: dict = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        return msg

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


@dataclass
class ToolResultMsg:
    tool_call_id: str
    name: str
    content: str
    role: str = "tool"

    def to_api(self) -> dict:
        return {
            "role": "tool", "tool_call_id": self.tool_call_id,
            "name": self.name, "content": self.content,
        }


Message = SystemMessage | UserMessage | AssistantMessage | ToolResultMsg


def messages_to_api(messages: list[Message]) -> list[dict]:
    return [m.to_api() for m in messages]
