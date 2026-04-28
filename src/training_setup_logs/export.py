"""Export redacted training units into trainer-friendly JSONL views."""

from __future__ import annotations

import json
from pathlib import Path

from training_setup_logs.pii import redact_jsonish
from training_setup_logs.schemas import LogEvent, TrainingUnit
from training_setup_logs.split import split_metadata
from training_setup_logs.tagging import tag_unit
from training_setup_logs.validate import validate_unit


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def to_sft_row(unit: TrainingUnit) -> dict[str, object]:
    """Convert a training unit to a LoRA-ready chat SFT row."""

    messages = _events_to_messages(unit.events)
    return {
        "id": unit.unit_id,
        "messages": messages,
        "metadata": {
            **unit.metadata,
            **split_metadata(unit),
            **tag_unit(unit),
            "validation_issues": [issue.to_dict() for issue in validate_unit(unit)],
        },
    }


def to_dpo_candidate_row(unit: TrainingUnit) -> dict[str, object] | None:
    """Create a DPO candidate row when feedback or error recovery gives a natural pair.

    This does not invent preference labels. It emits candidate pairs only when the
    session contains a failed assistant/error suffix followed by a later assistant
    recovery. Mentors can later approve, discard, or replace these pairs.
    """

    last_user_index = _last_index(unit.events, "user")
    if last_user_index is None:
        return None

    rejected = _first_event_after(unit.events, last_user_index, {"error"})
    chosen = _last_event_of_type(unit.events, "assistant")
    if rejected is None or chosen is None or chosen.event_id == rejected.event_id:
        return None

    prompt_events = unit.events[: last_user_index + 1]
    return {
        "id": f"{unit.unit_id}_dpo_candidate",
        "prompt": _events_to_messages(prompt_events),
        "chosen": [_event_to_message(chosen)],
        "rejected": [_event_to_message(rejected)],
        "metadata": {
            **split_metadata(unit),
            **tag_unit(unit),
            "source_unit_id": unit.unit_id,
            "requires_human_approval": True,
        },
    }


def to_redacted_unit_row(unit: TrainingUnit) -> dict[str, object]:
    """Return the canonical redacted unit for audit and downstream transforms."""

    return {
        **unit.to_dict(),
        "metadata": {
            **unit.metadata,
            **split_metadata(unit),
            **tag_unit(unit),
            "validation_issues": [issue.to_dict() for issue in validate_unit(unit)],
        },
    }


def _events_to_messages(events: list[LogEvent]) -> list[dict[str, object]]:
    messages: list[dict[str, object]] = []
    for event in events:
        message = _event_to_message(event)
        if message is not None:
            messages.append(message)
    return messages


def _event_to_message(event: LogEvent) -> dict[str, object] | None:
    if event.type in {"system", "user", "assistant"}:
        return {"role": event.type, "content": event.content or ""}
    if event.type == "tool_call":
        return {
            "role": "assistant",
            "content": json.dumps(
                {
                    "tool_call": {
                        "name": event.tool_name,
                        "arguments": event.tool_args or {},
                    }
                },
                sort_keys=True,
            ),
        }
    if event.type == "tool_result":
        return {"role": "tool", "name": event.tool_name or "unknown", "content": redact_jsonish(event.tool_result) or ""}
    if event.type == "error":
        return {"role": "assistant", "content": f"[ERROR] {event.content or redact_jsonish(event.tool_result) or ''}"}
    return None


def _last_index(events: list[LogEvent], event_type: str) -> int | None:
    for index in range(len(events) - 1, -1, -1):
        if events[index].type == event_type:
            return index
    return None


def _first_event_after(events: list[LogEvent], start_index: int, event_types: set[str]) -> LogEvent | None:
    for event in events[start_index + 1 :]:
        if event.type in event_types:
            return event
    return None


def _last_event_of_type(events: list[LogEvent], event_type: str) -> LogEvent | None:
    for event in reversed(events):
        if event.type == event_type:
            return event
    return None
