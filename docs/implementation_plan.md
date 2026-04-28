# Implementation Plan

## Goal

Build a privacy-safe, repeatable pipeline that converts production Q&A and agentic setup logs into validated SFT and DPO-ready training artifacts.

## Phase 1: End-to-end Prototype

- Normalize JSON/JSONL logs into the canonical `LogEvent` schema.
- Redact common PII and secrets with deterministic placeholders.
- Group sessions into Q&A and agent trajectory training units.
- Export LoRA-ready SFT rows and human-review DPO candidate rows.
- Produce a manifest with PII counts, validation counts, and complexity distribution.
- Assign deterministic splits and leakage buckets from normalized user text and tool names.

Status: initial stdlib-only prototype implemented.

## Phase 2: Quality and Safety

- Expand PII detection with organization-specific dictionaries.
- Add audit sampling reports with examples, false-negative review, and residual-risk notes.
- Add near-duplicate split checks to prevent leakage across SFT and DPO sets.
- Add stricter tool trajectory validation using declared tool schemas.

## Phase 3: Trainer Compatibility

- Add chat-template adapters for selected trainer stacks.
- Validate SFT export with a toy LoRA dry run.
- Validate DPO export with a small DPO dry run after human approval of candidate pairs.
- Add optional Hugging Face `datasets` integration while keeping JSONL as the primary interchange format.

## Phase 4: Student Model Path

- Filter trajectories by smaller context limits and supported tool sets.
- Build held-out behavioral checks for Q&A, tool use, recovery, and safety.
- Compare teacher and student outputs on the same redacted evaluation units.

## Open Mentor Questions

- What are the expected raw log formats and the minimum fields available per event?
- Which PII classes are in scope beyond emails, phone numbers, IDs, secrets, and URLs?
- Is there an approved source for preference labels, such as thumbs, corrections, or mentor-reviewed failed/successful traces?
- Which chat template and trainer stack should be treated as canonical for the first LoRA and DPO dry runs?
- What tool schemas or API definitions can be used to validate agent trajectories?
