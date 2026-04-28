"""Audit helpers for privacy review and data quality checks."""

from __future__ import annotations

from training_setup_logs.schemas import TrainingUnit
from training_setup_logs.validate import validate_unit


def build_audit_sample(units: list[TrainingUnit], limit: int = 10) -> list[dict[str, object]]:
    """Return a compact redacted sample for human audit workflows."""

    sample: list[dict[str, object]] = []
    for unit in units[:limit]:
        sample.append(
            {
                "unit_id": unit.unit_id,
                "session_id": unit.session_id,
                "unit_type": unit.unit_type,
                "event_count": len(unit.events),
                "preview": _preview(unit),
                "validation_issues": [issue.to_dict() for issue in validate_unit(unit)],
            }
        )
    return sample


def _preview(unit: TrainingUnit) -> list[dict[str, str]]:
    preview: list[dict[str, str]] = []
    for event in unit.events[:6]:
        text = event.content
        if text is None and event.tool_name:
            text = f"{event.type}: {event.tool_name}"
        if text is None and event.tool_result is not None:
            text = f"{event.type}: {type(event.tool_result).__name__}"
        preview.append(
            {
                "event_id": event.event_id,
                "type": event.type,
                "text": (text or "")[:180],
            }
        )
    return preview
