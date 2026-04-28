"""Command line interface for the log-to-training pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training_setup_logs.export import to_dpo_candidate_row, to_sft_row, write_jsonl
from training_setup_logs.ingest import load_events
from training_setup_logs.pii import PiiRedactor
from training_setup_logs.segment import segment_events
from training_setup_logs.tagging import tag_unit
from training_setup_logs.validate import validate_unit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build privacy-safe SFT and DPO JSONL from logs.")
    parser.add_argument("input", type=Path, help="Input JSON or JSONL log file.")
    parser.add_argument("--out-dir", type=Path, default=Path("out"), help="Output directory.")
    return parser


def run(input_path: Path, out_dir: Path) -> dict[str, object]:
    events = load_events(input_path)
    redactor = PiiRedactor()
    redacted_events = [redactor.redact_event(event) for event in events]
    units = segment_events(redacted_events)

    sft_rows = [to_sft_row(unit) for unit in units]
    dpo_rows = [row for unit in units if (row := to_dpo_candidate_row(unit)) is not None]
    manifest = {
        "input": str(input_path),
        "unit_count": len(units),
        "sft_rows": len(sft_rows),
        "dpo_candidate_rows": len(dpo_rows),
        "pii_counts": redactor.report.counts_by_kind(),
        "validation_issue_count": sum(len(validate_unit(unit)) for unit in units),
        "tag_summary": _tag_summary(units),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "sft.jsonl", sft_rows)
    write_jsonl(out_dir / "dpo_candidates.jsonl", dpo_rows)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main() -> None:
    args = build_parser().parse_args()
    manifest = run(args.input, args.out_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))


def _tag_summary(units: list[object]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for unit in units:
        complexity = str(tag_unit(unit)["complexity"])
        summary[complexity] = summary.get(complexity, 0) + 1
    return summary


if __name__ == "__main__":
    main()
