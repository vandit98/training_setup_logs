# Canonical Pipeline Schemas

This prototype uses a small canonical event schema so heterogeneous product logs can be normalized before redaction, segmentation, tagging, validation, and export.

## `LogEvent`

| Field | Meaning |
| --- | --- |
| `event_id` | Stable event identifier. Generated when missing. |
| `session_id` | Conversation, trace, or request group identifier. |
| `timestamp` | Optional event timestamp. |
| `type` | One of `system`, `user`, `assistant`, `tool_call`, `tool_result`, `error`, `feedback`. |
| `content` | Natural-language text for chat, errors, or comments. |
| `tool_name` | Tool/API name for tool calls and results when available. |
| `tool_args` | JSON object passed to a tool. Redacted recursively. |
| `tool_result` | Tool observation/result. Redacted recursively. |
| `metadata` | Non-canonical fields preserved from source logs. |

## `TrainingUnit`

Events are grouped by `session_id`. A unit with any `tool_call`, `tool_result`, or `error` event is tagged as `agent_trajectory`; otherwise it is tagged as `qa`.

Each unit receives:

- trajectory complexity tags: `single_turn_qa`, `multi_turn_qa`, `single_tool`, `multi_tool`, `recovery`
- scheduling bucket: foundation, tool-use, or complex trajectories
- validation issues for missing tool names, orphan tool results, and empty text turns
- SFT JSONL row with a `messages` array
- optional DPO candidate row when an error/failure and later assistant recovery create a natural preference candidate

## Privacy Defaults

The initial redactor is deterministic within a run and replaces detected sensitive spans with placeholders such as `<EMAIL_1>`, `<PHONE_1>`, and `<API_KEY_1>`. It covers common emails, phone numbers, Aadhaar-like IDs, IP addresses, bearer tokens, API-key-shaped secrets, and URL secret query parameters.

This is intentionally a baseline. Production use should add organization-specific dictionaries, human audit sampling, residual-risk reporting, and policy approval before any data reaches a training store.
