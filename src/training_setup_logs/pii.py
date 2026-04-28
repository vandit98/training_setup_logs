"""Deterministic rule-based PII redaction."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from training_setup_logs.schemas import LogEvent, PiiFinding


PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("bearer_token", re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{16,}\b", re.IGNORECASE)),
    ("api_key", re.compile(r"\b(?:sk|pk|api|key|token|secret)[-_]?[A-Za-z0-9]{16,}\b", re.IGNORECASE)),
    ("url_token", re.compile(r"([?&](?:token|key|secret|signature|auth)=)[^&\s]+", re.IGNORECASE)),
    ("email", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)),
    ("aadhaar", re.compile(r"(?<!\d)(?:\d{4}[\s-]?){2}\d{4}(?!\d)")),
    ("phone", re.compile(r"(?<![A-Za-z0-9_])(?:\+?\d[\d\s().-]{8,}\d)(?![A-Za-z0-9_])")),
    ("ip_address", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
]


@dataclass
class RedactionReport:
    """Redaction findings accumulated for a pipeline run."""

    findings: list[PiiFinding] = field(default_factory=list)

    def add(self, kind: str, placeholder: str) -> None:
        self.findings.append(PiiFinding(kind=kind, placeholder=placeholder))

    def counts_by_kind(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for finding in self.findings:
            counts[finding.kind] = counts.get(finding.kind, 0) + 1
        return counts


class PiiRedactor:
    """Replace sensitive spans with stable placeholders within one run."""

    def __init__(self) -> None:
        self._seen: dict[tuple[str, str], str] = {}
        self.report = RedactionReport()

    def redact_text(self, text: str | None) -> str | None:
        if text is None:
            return None
        redacted = text
        for kind, pattern in PII_PATTERNS:
            redacted = pattern.sub(lambda match: self._placeholder(kind, match.group(0)), redacted)
        return redacted

    def redact_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.redact_text(value)
        if isinstance(value, dict):
            return {key: self.redact_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.redact_value(item) for item in value]
        return value

    def redact_event(self, event: LogEvent) -> LogEvent:
        metadata = dict(event.metadata)
        metadata["pii_redacted"] = True
        return LogEvent(
            event_id=event.event_id,
            session_id=event.session_id,
            timestamp=event.timestamp,
            type=event.type,
            content=self.redact_text(event.content),
            tool_name=event.tool_name,
            tool_args=self.redact_value(event.tool_args),
            tool_result=self.redact_value(event.tool_result),
            metadata=metadata,
        )

    def _placeholder(self, kind: str, raw: str) -> str:
        if kind == "url_token":
            prefix = raw.split("=", 1)[0] + "="
            secret = raw.split("=", 1)[1] if "=" in raw else raw
            return prefix + self._placeholder("url_secret", secret)

        key = (kind, raw)
        if key not in self._seen:
            placeholder = f"<{kind.upper()}_{len([k for k in self._seen if k[0] == kind]) + 1}>"
            self._seen[key] = placeholder
            self.report.add(kind, placeholder)
        return self._seen[key]


def redact_jsonish(value: Any) -> str | None:
    """Return a stable string representation for redacted tool payloads."""

    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, default=str)
