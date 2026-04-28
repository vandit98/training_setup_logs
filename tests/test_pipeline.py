from pathlib import Path

from training_setup_logs.cli import run
from training_setup_logs.ingest import load_events
from training_setup_logs.pii import PiiRedactor
from training_setup_logs.schemas import LogEvent, TrainingUnit
from training_setup_logs.segment import segment_events
from training_setup_logs.split import split_metadata
from training_setup_logs.tagging import tag_unit
from training_setup_logs.tool_schema import ToolRegistry
from training_setup_logs.validate import validate_unit


ROOT = Path(__file__).resolve().parents[1]


def test_sample_pipeline_exports_sft_and_dpo_candidates(tmp_path):
    manifest = run(ROOT / "examples" / "sample_agent_logs.jsonl", tmp_path)

    assert manifest["unit_count"] == 2
    assert manifest["sft_rows"] == 2
    assert manifest["dpo_candidate_rows"] == 1
    assert manifest["pii_counts"]["email"] == 1
    assert manifest["pii_counts"]["phone"] == 1
    assert manifest["validation_issue_count"] == 0
    assert (tmp_path / "sft.jsonl").exists()
    assert (tmp_path / "dpo_candidates.jsonl").exists()
    assert (tmp_path / "redacted_units.jsonl").exists()
    assert (tmp_path / "redaction_report.json").exists()
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


def test_split_metadata_keeps_similar_prompts_together():
    first = TrainingUnit(
        unit_id="u1",
        session_id="s1",
        unit_type="qa",
        events=[
            LogEvent(event_id="e1", session_id="s1", timestamp=None, type="user", content="  What is LoRA?  "),
            LogEvent(event_id="e2", session_id="s1", timestamp=None, type="assistant", content="Answer"),
        ],
    )
    second = TrainingUnit(
        unit_id="u2",
        session_id="s2",
        unit_type="qa",
        events=[
            LogEvent(event_id="e3", session_id="s2", timestamp=None, type="user", content="what   is lora?"),
            LogEvent(event_id="e4", session_id="s2", timestamp=None, type="assistant", content="Answer"),
        ],
    )

    assert split_metadata(first) == split_metadata(second)


def test_validation_flags_tool_mismatch_and_missing_observation():
    mismatch_unit = TrainingUnit(
        unit_id="u3",
        session_id="s3",
        unit_type="agent_trajectory",
        events=[
            LogEvent(event_id="e1", session_id="s3", timestamp=None, type="tool_call", tool_name="weather"),
            LogEvent(event_id="e2", session_id="s3", timestamp=None, type="tool_result", tool_name="crop"),
        ],
    )
    missing_result_unit = TrainingUnit(
        unit_id="u4",
        session_id="s4",
        unit_type="agent_trajectory",
        events=[
            LogEvent(event_id="e3", session_id="s4", timestamp=None, type="tool_call", tool_name="weather"),
        ],
    )

    assert [issue.code for issue in validate_unit(mismatch_unit)] == ["TOOL_RESULT_NAME_MISMATCH"]
    assert [issue.code for issue in validate_unit(missing_result_unit)] == ["MISSING_TOOL_OBSERVATION"]


def test_tool_schema_validation_flags_unknown_tools_and_missing_args():
    registry = ToolRegistry.from_path(ROOT / "examples" / "tool_schema.json")
    unit = TrainingUnit(
        unit_id="u5",
        session_id="s5",
        unit_type="agent_trajectory",
        events=[
            LogEvent(
                event_id="e1",
                session_id="s5",
                timestamp=None,
                type="tool_call",
                tool_name="get_weather",
                tool_args={},
            ),
            LogEvent(
                event_id="e2",
                session_id="s5",
                timestamp=None,
                type="tool_result",
                tool_name="get_weather",
                tool_result={},
            ),
            LogEvent(
                event_id="e3",
                session_id="s5",
                timestamp=None,
                type="tool_call",
                tool_name="unknown_tool",
            ),
            LogEvent(
                event_id="e4",
                session_id="s5",
                timestamp=None,
                type="error",
                content="failed",
            ),
        ],
    )

    codes = [issue.code for issue in validate_unit(unit, registry)]

    assert "MISSING_TOOL_ARGS" in codes
    assert "UNKNOWN_TOOL" in codes
