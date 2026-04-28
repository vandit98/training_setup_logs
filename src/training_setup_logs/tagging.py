"""Trajectory complexity and scheduling metadata."""

from __future__ import annotations

from training_setup_logs.schemas import TrainingUnit


AMBIGUITY_MARKERS = ("maybe", "not sure", "unclear", "ambiguous", "try again", "fallback")


def tag_unit(unit: TrainingUnit) -> dict[str, object]:
    """Compute training-time scheduling tags for a unit."""

    tool_calls = [event for event in unit.events if event.type == "tool_call"]
    errors = [event for event in unit.events if event.type == "error"]
    assistant_turns = [event for event in unit.events if event.type == "assistant"]
    user_turns = [event for event in unit.events if event.type == "user"]
    unique_tools = sorted({event.tool_name for event in tool_calls if event.tool_name})
    has_recovery = _has_recovery_after_error(unit)
    has_ambiguity = any(
        marker in (event.content or "").lower()
        for event in unit.events
        for marker in AMBIGUITY_MARKERS
    )

    complexity = _complexity(
        event_count=len(unit.events),
        tool_count=len(tool_calls),
        error_count=len(errors),
        has_recovery=has_recovery,
    )

    return {
        "unit_type": unit.unit_type,
        "complexity": complexity,
        "event_count": len(unit.events),
        "user_turns": len(user_turns),
        "assistant_turns": len(assistant_turns),
        "tool_call_count": len(tool_calls),
        "unique_tools": unique_tools,
        "error_count": len(errors),
        "has_recovery": has_recovery,
        "has_ambiguity": has_ambiguity,
        "recommended_schedule_bucket": _schedule_bucket(complexity),
    }


def _has_recovery_after_error(unit: TrainingUnit) -> bool:
    seen_error = False
    for event in unit.events:
        if event.type == "error":
            seen_error = True
        elif seen_error and event.type in {"assistant", "tool_call"}:
            return True
    return False


def _complexity(event_count: int, tool_count: int, error_count: int, has_recovery: bool) -> str:
    if has_recovery or error_count:
        return "recovery"
    if tool_count >= 2 or event_count >= 8:
        return "multi_tool"
    if tool_count == 1:
        return "single_tool"
    if event_count > 2:
        return "multi_turn_qa"
    return "single_turn_qa"


def _schedule_bucket(complexity: str) -> str:
    if complexity in {"single_turn_qa", "multi_turn_qa"}:
        return "phase_1_foundation"
    if complexity == "single_tool":
        return "phase_2_tool_use"
    return "phase_3_complex_trajectories"
