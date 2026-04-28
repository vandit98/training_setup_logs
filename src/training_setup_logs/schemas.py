"""Canonical schemas for normalized logs and training units."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

EventType = Literal[
    "system",
    "user",
    "assistant",
    "tool_call",
    "tool_result",
    "error",
    "feedback",
]

UnitType = Literal["qa", "agent_trajectory"]


@dataclass(frozen=True)
class PiiFinding:
    """A single PII finding after redaction."""

    kind: str
    placeholder: str

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "placeholder": self.placeholder}


@dataclass(frozen=True)
class LogEvent:
    """A normalized event from application, chat, or agent logs."""

    event_id: str
    session_id: str
    timestamp: str | None
    type: EventType
    content: str | None = None
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_result: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "type": self.type,
            "content": self.content,
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "tool_result": self.tool_result,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class TrainingUnit:
    """A redacted session segment ready for tagging and export."""

    unit_id: str
    session_id: str
    unit_type: UnitType
    events: list[LogEvent]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "session_id": self.session_id,
            "unit_type": self.unit_type,
            "events": [event.to_dict() for event in self.events],
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class ValidationIssue:
    """Validation issue attached to a training unit."""

    code: str
    message: str
    event_id: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {"code": self.code, "message": self.message, "event_id": self.event_id}
