"""Session segmentation into Q&A and agent trajectory training units."""

from __future__ import annotations

import hashlib
from collections import defaultdict

from training_setup_logs.schemas import LogEvent, TrainingUnit


def segment_events(events: list[LogEvent]) -> list[TrainingUnit]:
    """Group normalized events into redacted training units by session."""

    sessions: dict[str, list[LogEvent]] = defaultdict(list)
    for event in events:
        sessions[event.session_id].append(event)

    units: list[TrainingUnit] = []
    for session_id, session_events in sorted(sessions.items()):
        unit_type = "agent_trajectory" if any(
            event.type in {"tool_call", "tool_result", "error"} for event in session_events
        ) else "qa"
        unit_id = _unit_id(session_id, session_events)
        units.append(
            TrainingUnit(
                unit_id=unit_id,
                session_id=session_id,
                unit_type=unit_type,
                events=session_events,
                metadata={"source_event_count": len(session_events)},
            )
        )
    return units


def _unit_id(session_id: str, events: list[LogEvent]) -> str:
    joined = "|".join(event.event_id for event in events)
    digest = hashlib.sha1(f"{session_id}:{joined}".encode("utf-8")).hexdigest()[:12]
    return f"unit_{digest}"
