"""Deterministic split assignment with simple leakage guards."""

from __future__ import annotations

import hashlib
import re

from training_setup_logs.schemas import TrainingUnit


SPACE_RE = re.compile(r"\s+")


def split_metadata(unit: TrainingUnit) -> dict[str, str]:
    """Return stable split metadata for a unit.

    The leakage bucket is based on normalized user text instead of unit id, so
    repeated or near-identical prompts are kept in the same split.
    """

    leakage_key = _leakage_key(unit)
    split = _split_from_key(leakage_key)
    return {
        "split": split,
        "leakage_bucket": hashlib.sha1(leakage_key.encode("utf-8")).hexdigest()[:16],
    }


def _leakage_key(unit: TrainingUnit) -> str:
    user_text = "\n".join(
        _normalize_text(event.content or "")
        for event in unit.events
        if event.type == "user"
    )
    tool_names = ",".join(
        sorted({event.tool_name or "unknown" for event in unit.events if event.type == "tool_call"})
    )
    return f"{unit.unit_type}|{tool_names}|{user_text}"


def _normalize_text(text: str) -> str:
    return SPACE_RE.sub(" ", text.casefold()).strip()


def _split_from_key(key: str) -> str:
    value = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16) % 100
    if value < 80:
        return "train"
    if value < 90:
        return "validation"
    return "test"
