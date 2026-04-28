from pathlib import Path

from training_setup_logs.cli import run
from training_setup_logs.ingest import load_events
from training_setup_logs.pii import PiiRedactor
from training_setup_logs.segment import segment_events
from training_setup_logs.tagging import tag_unit
from training_setup_logs.validate import validate_unit


ROOT = Path(__file__).resolve().parents[1]


def test_sample_pipeline_exports_sft_and_dpo_candidates(tmp_path):
    manifest = run(ROOT / "examples" / "sample_agent_logs.jsonl", tmp_path)

    assert manifest["unit_count"] == 2
    assert manifest["sft_rows"] == 2
    assert manifest["dpo_candidate_rows"] == 1
    assert manifest["pii_counts"]["email"] == 1
    assert manifest["pii_counts"]["phone"] == 1
    assert (tmp_path / "sft.jsonl").exists()
    assert (tmp_path / "dpo_candidates.jsonl").exists()
    assert (tmp_path / "manifest.json").exists()


def test_redaction_is_deterministic_within_run():
    events = load_events(ROOT / "examples" / "sample_agent_logs.jsonl")
    redactor = PiiRedactor()
    redacted = [redactor.redact_event(event) for event in events]

    assert "<EMAIL_1>" in (redacted[0].content or "")
    assert redacted[4].tool_args["phone"] == "<PHONE_1>"
    assert redacted[2].tool_args["api_key"] == "<API_KEY_1>"


def test_agent_trace_receives_recovery_tag_and_validates():
    events = load_events(ROOT / "examples" / "sample_agent_logs.jsonl")
    redactor = PiiRedactor()
    units = segment_events([redactor.redact_event(event) for event in events])
    agent_unit = next(unit for unit in units if unit.unit_type == "agent_trajectory")

    tags = tag_unit(agent_unit)

    assert tags["complexity"] == "recovery"
    assert tags["recommended_schedule_bucket"] == "phase_3_complex_trajectories"
    assert validate_unit(agent_unit) == []
