"""Input parsers for heterogeneous JSON/JSONL logs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from training_setup_logs.schemas import EventType, LogEvent


TYPE_ALIASES: dict[str, EventType] = {
    "human": "user",
    "user": "user",
    "ai": "assistant",
    "assistant": "assistant",
    "bot": "assistant",
    "system": "system",
    "tool": "tool_call",
    "tool_call": "tool_call",
    "function_call": "tool_call",
    "tool_result": "tool_result",
    "observation": "tool_result",
    "error": "error",
    "exception": "error",
    "feedback": "feedback",
}


def read_json_records(path: Path) -> list[dict[str, Any]]:
    """Read JSON, JSONL, or a JSON object containing an events/logs array."""

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    payload = json.loads(text)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("events", "logs", "records"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        return [payload]
    raise ValueError(f"Unsupported JSON payload in {path}")


def normalize_records(records: Iterable[dict[str, Any]]) -> list[LogEvent]:
    """Normalize raw records into the canonical LogEvent schema."""

    events: list[LogEvent] = []
    for index, record in enumerate(records):
        session_id = str(
            record.get("session_id")
            or record.get("conversation_id")
            or record.get("thread_id")
            or "default"
        )
        raw_type = str(record.get("type") or record.get("role") or record.get("event") or "assistant")
        event_type = TYPE_ALIASES.get(raw_type.lower())
        if event_type is None:
            event_type = "assistant"

        tool_name = record.get("tool_name") or record.get("name")
        if event_type == "tool_call" and not tool_name and isinstance(record.get("tool"), str):
            tool_name = record["tool"]

        event_id = str(record.get("event_id") or _stable_event_id(record, index))
        events.append(
            LogEvent(
                event_id=event_id,
                session_id=session_id,
                timestamp=record.get("timestamp") or record.get("time") or record.get("created_at"),
                type=event_type,
                content=_string_or_none(record.get("content") or record.get("message") or record.get("text")),
                tool_name=_string_or_none(tool_name),
                tool_args=_dict_or_none(record.get("tool_args") or record.get("arguments") or record.get("args")),
                tool_result=record.get("tool_result") or record.get("result") or record.get("observation"),
                metadata=_metadata_without_known_fields(record),
            )
        )
    return sorted(events, key=lambda event: (event.session_id, event.timestamp or "", event.event_id))


def load_events(path: Path) -> list[LogEvent]:
    """Load and normalize records from one JSON or JSONL file."""

    return normalize_records(read_json_records(path))


def _stable_event_id(record: dict[str, Any], index: int) -> str:
    payload = json.dumps(record, sort_keys=True, default=str)
    digest = hashlib.sha1(f"{index}:{payload}".encode("utf-8")).hexdigest()[:12]
    return f"evt_{digest}"


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, default=str)


def _dict_or_none(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    return None


def _metadata_without_known_fields(record: dict[str, Any]) -> dict[str, Any]:
    known = {
        "session_id",
        "conversation_id",
        "thread_id",
        "type",
        "role",
        "event",
        "timestamp",
        "time",
        "created_at",
        "content",
        "message",
        "text",
        "tool_name",
        "tool",
        "name",
        "tool_args",
        "arguments",
        "args",
        "tool_result",
        "result",
        "observation",
        "event_id",
    }
    return {key: value for key, value in record.items() if key not in known}
