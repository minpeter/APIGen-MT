from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

from apigen_step_by_step import (
    ConversationTrajectory,
    StepByStepDatapoint,
    ToolCallWithOutput,
    TrajectoryStep,
)
from llm_request_helpers import (
    decode_json_object,
    is_object_list,
    require_object_map,
)
from tool_manager import ToolManager

PROJECT_ROOT = Path(__file__).parents[2]


def build_trajectory_fixture(
    tmp_path: Path,
    *,
    tampered: bool,
    unavailable: bool = False,
) -> tuple[Path, Path, Path]:
    tool_pool = tmp_path / "tools.jsonl"
    invocation_examples = tmp_path / "invocations.jsonl"
    trajectory_path = tmp_path / "trajectory.json"
    _ = tool_pool.write_text(
        json.dumps(
            {
                "api_name": "get_user_id",
                "tool_name": "message_api",
                "api_description": "Get a workspace user identifier",
                "category": "Communication",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user": {
                            "type": "string",
                            "description": "User name",
                        }
                    },
                    "required": ["user"],
                },
                "output_type": "dict",
                "output_description": "User identifier",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _ = invocation_examples.write_text(
        json.dumps(
            {
                "initial_config": {
                    "MessageAPI": {
                        "workspace_id": "WS001",
                        "user_count": 1,
                        "user_map": {"Alice": "USR001"},
                        "messages_sent_map": {"USR001": {}},
                        "messages_inbox_map": {"USR001": {}},
                        "message_count": 0,
                        "current_user": "USR001",
                    }
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manager = ToolManager(
        None,
        str(tool_pool),
        invocation_examples_path=str(invocation_examples),
        use_config_pool=False,
    )
    initial_state = manager.get_api_state()
    output = manager.invoke_python_tool("get_user_id", {"user": "Alice"})
    post_state = manager.get_api_state()
    recorded_output = require_object_map(
        copy.deepcopy(output),
        source="fixture tool output",
    )
    if tampered:
        recorded_output["user_id"] = "USR999"
    step = TrajectoryStep(
        step_number=1,
        tool_calls=[
            ToolCallWithOutput(
                tool_name="remote_tool" if unavailable else "get_user_id",
                arguments={"user": "Alice"},
                output=recorded_output,
            )
        ],
        pre_state=initial_state,
        post_state=post_state,
    )
    datapoint = StepByStepDatapoint(
        trajectory=ConversationTrajectory(
            query="Find Alice's workspace user identifier",
            steps=[step],
            final_response="Alice is USR001",
            tools_used=["get_user_id"],
            categories_used=["Communication"],
            initial_api_state=initial_state,
        ),
        generation_metadata={"num_actions": 1},
    )
    _ = trajectory_path.write_text(
        datapoint.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return trajectory_path, tool_pool, invocation_examples


def run_verification_cli(
    tmp_path: Path,
    *,
    tampered: bool,
    unavailable: bool = False,
) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
    trajectory, tool_pool, invocations = build_trajectory_fixture(
        tmp_path,
        tampered=tampered,
        unavailable=unavailable,
    )
    output_path = tmp_path / "verification.json"
    env = os.environ.copy()
    _ = env.pop("OPENAI_API_KEY", None)
    _ = env.pop("OPENAI_API_BASE", None)
    result = subprocess.run(
        [
            sys.executable,
            "main.py",
            "--verify-trajectory",
            str(trajectory),
            "--verification-output",
            str(output_path),
            "--tool-pool",
            str(tool_pool),
            "--invocation-examples",
            str(invocations),
        ],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert output_path.exists(), result.stderr
    verification = decode_json_object(
        output_path.read_text(encoding="utf-8"),
        source="verification output",
    )
    return result, verification


def test_cli_accepts_replayable_trajectory(tmp_path: Path):
    result, verification = run_verification_cli(tmp_path, tampered=False)
    assert result.returncode == 0
    assert verification["status"] == "verified"
    assert verification["is_valid"] is True
    assert verification["checked_calls"] == 1


def test_cli_rejects_tampered_trajectory(tmp_path: Path):
    result, verification = run_verification_cli(tmp_path, tampered=True)
    assert result.returncode == 1
    assert verification["status"] == "failed"
    assert verification["is_valid"] is False
    issues = verification["issues"]
    assert is_object_list(issues)
    issue = require_object_map(issues[0], source="verification issue")
    assert issue["check"] == "output"


def test_cli_reports_unavailable_trajectory(tmp_path: Path):
    result, verification = run_verification_cli(
        tmp_path,
        tampered=False,
        unavailable=True,
    )
    assert result.returncode == 2
    assert verification["status"] == "unavailable"
    assert verification["is_valid"] is False
    assert verification["unavailable_tools"] == ["remote_tool"]


def test_cli_rejects_malformed_trajectory(tmp_path: Path):
    trajectory, tool_pool, invocations = build_trajectory_fixture(
        tmp_path,
        tampered=False,
    )
    _ = trajectory.write_text("{}", encoding="utf-8")
    output_path = tmp_path / "verification.json"
    result = subprocess.run(
        [
            sys.executable,
            "main.py",
            "--verify-trajectory",
            str(trajectory),
            "--verification-output",
            str(output_path),
            "--tool-pool",
            str(tool_pool),
            "--invocation-examples",
            str(invocations),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert not output_path.exists()
    assert "ERROR:" in result.stderr
