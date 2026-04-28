"""Schema and trajectory validation."""

from __future__ import annotations

from training_setup_logs.schemas import TrainingUnit, ValidationIssue
from training_setup_logs.tool_schema import ToolRegistry


def validate_unit(unit: TrainingUnit, tool_registry: ToolRegistry | None = None) -> list[ValidationIssue]:
    """Validate basic event and tool trajectory consistency."""

    issues: list[ValidationIssue] = []
    pending_tools: list[tuple[str, str]] = []
    registry = tool_registry or ToolRegistry({})

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
                pending_tools.append((event.tool_name, event.event_id))
                if registry.tools and not registry.has_tool(event.tool_name):
                    issues.append(
                        ValidationIssue(
                            code="UNKNOWN_TOOL",
                            message=f"Tool {event.tool_name} is not declared in the registry.",
                            event_id=event.event_id,
                        )
                    )
                missing_args = registry.missing_required_args(event.tool_name, event.tool_args)
                if missing_args:
                    issues.append(
                        ValidationIssue(
                            code="MISSING_TOOL_ARGS",
                            message=f"Tool {event.tool_name} is missing required args: {sorted(missing_args)}.",
                            event_id=event.event_id,
                        )
                    )

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
                expected_tool, _ = pending_tools.pop(0)
                if event.tool_name and event.tool_name != expected_tool:
                    issues.append(
                        ValidationIssue(
                            code="TOOL_RESULT_NAME_MISMATCH",
                            message=f"Tool result is for {event.tool_name}, expected {expected_tool}.",
                            event_id=event.event_id,
                        )
                    )

        if event.type == "error" and pending_tools:
            pending_tools.pop(0)

        if event.type in {"user", "assistant"} and not event.content:
            issues.append(
                ValidationIssue(
                    code="EMPTY_TEXT_TURN",
                    message=f"{event.type} event has no content.",
                    event_id=event.event_id,
                )
            )

    for tool_name, event_id in pending_tools:
        issues.append(
            ValidationIssue(
                code="MISSING_TOOL_OBSERVATION",
                message=f"Tool call {tool_name} has no following result or error event.",
                event_id=event_id,
            )
        )

    return issues
