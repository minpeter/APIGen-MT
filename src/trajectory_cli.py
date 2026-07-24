"""Deterministic verification for trajectory JSON and JSONL files."""

from __future__ import annotations

import json
from pathlib import Path

from llm_request_helpers import (
    decode_json_object,
    is_object_list,
    parse_json_output,
    require_object_map,
)
from multi_turn_models import MultiTurnDatapoint
from step_by_step_models import ConversationTrajectory, StepByStepDatapoint
from trajectory_replay import verify_trajectory_replay

type VerificationSummary = dict[str, object]


def _load_payloads(path: Path) -> list[dict[str, object]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [
            decode_json_object(line, source="trajectory record")
            for line in text.splitlines()
            if line.strip()
        ]

    payload = parse_json_output(text)
    if not is_object_list(payload):
        return [require_object_map(payload, source="trajectory")]
    return [require_object_map(item, source="trajectory record") for item in payload]


def _result_summary(result_json: str) -> VerificationSummary:
    return decode_json_object(result_json, source="verification result")


def _verify_payload(
    payload: dict[str, object],
    manager: object,
) -> VerificationSummary:
    if "trajectory" in payload:
        datapoint = StepByStepDatapoint.model_validate(payload)
        result = verify_trajectory_replay(
            manager,
            datapoint.trajectory.steps,
            datapoint.trajectory.initial_api_state,
        )
        return _result_summary(result.model_dump_json())

    if "conversation" in payload:
        datapoint = MultiTurnDatapoint.model_validate(payload)
        steps = [step for turn in datapoint.conversation.turns for step in turn.steps]
        result = verify_trajectory_replay(
            manager,
            steps,
            datapoint.initial_api_state or datapoint.conversation.initial_api_state,
        )
        return _result_summary(result.model_dump_json())

    trajectory = ConversationTrajectory.model_validate(payload)
    result = verify_trajectory_replay(
        manager,
        trajectory.steps,
        trajectory.initial_api_state,
    )
    return _result_summary(result.model_dump_json())


def _status(summary: VerificationSummary) -> str:
    status = summary.get("status")
    if not isinstance(status, str):
        raise TypeError("Verification result has no status")
    return status


def _is_valid(summary: VerificationSummary) -> bool:
    is_valid = summary.get("is_valid")
    if not isinstance(is_valid, bool):
        raise TypeError("Verification result has no validity flag")
    return is_valid


def verify_trajectory_file(
    input_path: str,
    output_path: str,
    manager: object,
) -> tuple[VerificationSummary, int]:
    """Verify every input record, persist the summary, and return its exit code."""
    results = [
        _verify_payload(payload, manager)
        for payload in _load_payloads(Path(input_path))
    ]
    if not results:
        raise ValueError("Trajectory file contains no records")

    summary: VerificationSummary
    if len(results) == 1:
        summary = results[0]
    else:
        failed = any(not _is_valid(result) for result in results)
        unavailable = any(_status(result) == "unavailable" for result in results)
        status = "failed" if failed else "unavailable" if unavailable else "verified"
        summary = {
            "status": status,
            "is_valid": not failed,
            "items": results,
        }

    _ = Path(output_path).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    status = _status(summary)
    if status == "failed":
        return summary, 1
    if status == "unavailable":
        return summary, 2
    return summary, 0
