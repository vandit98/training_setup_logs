"""Schema and trajectory validation."""

from __future__ import annotations

from training_setup_logs.schemas import TrainingUnit, ValidationIssue


def validate_unit(unit: TrainingUnit) -> list[ValidationIssue]:
    """Validate basic event and tool trajectory consistency."""

    issues: list[ValidationIssue] = []
    pending_tools: list[str] = []

    for event in unit.events:
        if event.type == "tool_call":
            if not event.tool_name:
                issues.append(
                    ValidationIssue(
                        code="TOOL_NAME_MISSING",
                        message="Tool call event is missing tool_name.",
                        event_id=event.event_id,
                    )
                )
            else:
                pending_tools.append(event.tool_name)

        if event.type == "tool_result":
            if not pending_tools:
                issues.append(
                    ValidationIssue(
                        code="ORPHAN_TOOL_RESULT",
                        message="Tool result appears before any matching tool call.",
                        event_id=event.event_id,
                    )
                )
            else:
                pending_tools.pop(0)

        if event.type in {"user", "assistant"} and not event.content:
            issues.append(
                ValidationIssue(
                    code="EMPTY_TEXT_TURN",
                    message=f"{event.type} event has no content.",
                    event_id=event.event_id,
                )
            )

    return issues
