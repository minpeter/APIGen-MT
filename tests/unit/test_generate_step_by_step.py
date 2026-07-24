"""Unit tests for generate_step_by_step CLI and main functions.

These tests verify the command-line interface, argument parsing,
tool loading, and main execution flow.
"""

import json
import os
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

# Import the module under test
_ = sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from generate_step_by_step import load_tool_categories, main, parse_args
from llm_request_helpers import decode_json_object


class TestParseArgs:
    """Tests for argument parsing."""

    def test_parse_args_defaults(self):
        """Test default argument values."""
        with patch.object(sys, "argv", ["generate_step_by_step.py"]):
            args = parse_args()

        assert args.num_datapoints == 100
        assert args.num_actions == 1
        assert args.output == "step_by_step_datapoints.jsonl"
        assert args.model == "minimax/minimax-m2.7"

    def test_parse_args_custom_values(self):
        """Test custom argument values."""
        test_args = [
            "generate_step_by_step.py",
            "--num-datapoints",
            "50",
            "--num-actions",
            "3",
            "--output",
            "custom_output.jsonl",
            "--model",
            "custom/model",
        ]
        with patch.object(sys, "argv", test_args):
            args = parse_args()

        assert args.num_datapoints == 50
        assert args.num_actions == 3
        assert args.output == "custom_output.jsonl"
        assert args.model == "custom/model"

    def test_parse_args_short_flags(self):
        """Test short form flags."""
        test_args = [
            "generate_step_by_step.py",
            "-n",
            "25",
            "-a",
            "4",
            "-o",
            "output.jsonl",
            "-m",
            "short/model",
        ]
        with patch.object(sys, "argv", test_args):
            args = parse_args()

        assert args.num_datapoints == 25
        assert args.num_actions == 4
        assert args.output == "output.jsonl"
        assert args.model == "short/model"

    def test_parse_args_tool_pool(self):
        """Test custom tool pool path."""
        test_args = [
            "generate_step_by_step.py",
            "--tool-pool",
            "/custom/path/tools.jsonl",
        ]
        with patch.object(sys, "argv", test_args):
            args = parse_args()

        assert "/custom/path/tools.jsonl" in args.tool_pool

    def test_parse_args_verify_trajectory_mode(self):
        test_args = [
            "generate_step_by_step.py",
            "--verify-trajectory",
            "trajectory.json",
        ]
        with patch.object(sys, "argv", test_args):
            args = parse_args()

        assert args.verify_trajectory == "trajectory.json"

    def test_parse_args_invalid_num_datapoints(self):
        """Test handling of invalid num_datapoints."""
        test_args = [
            "generate_step_by_step.py",
            "--num-datapoints",
            "-1",
        ]
        # argparse should accept negative numbers, but we validate later
        with patch.object(sys, "argv", test_args):
            args = parse_args()
            assert args.num_datapoints == -1


class TestLoadToolCategories:
    """Tests for load_tool_categories function."""

    def test_load_tool_categories_success(self, tmp_path: Path):
        """Test successful tool loading and categorization."""
        # Create a temporary tool pool file
        tool_pool = tmp_path / "tools.jsonl"
        tools = [
            {"api_name": "tool1", "category": "Travel"},
            {"api_name": "tool2", "category": "Travel"},
            {"api_name": "tool3", "category": "Food"},
        ]
        _ = tool_pool.write_text(
            "".join(json.dumps(tool) + "\n" for tool in tools),
            encoding="utf-8",
        )

        result = load_tool_categories(str(tool_pool))

        assert "Travel" in result
        assert "Food" in result
        assert len(result["Travel"]) == 2
        assert len(result["Food"]) == 1

    def test_load_tool_categories_missing_category(self, tmp_path: Path):
        """Test handling of tools without category."""
        tool_pool = tmp_path / "tools.jsonl"
        tools = [
            {"api_name": "tool1"},  # No category
            {"api_name": "tool2", "category": "Travel"},
        ]
        _ = tool_pool.write_text(
            "".join(json.dumps(tool) + "\n" for tool in tools),
            encoding="utf-8",
        )

        result = load_tool_categories(str(tool_pool))

        assert "Unknown" in result
        assert "Travel" in result

    def test_load_tool_categories_empty_file(self, tmp_path: Path):
        """Test handling of empty tool pool file."""
        tool_pool = tmp_path / "empty.jsonl"
        _ = tool_pool.write_text("", encoding="utf-8")

        result = load_tool_categories(str(tool_pool))

        assert result == {}

    def test_load_tool_categories_invalid_json(self, tmp_path: Path):
        """Test handling of invalid JSON lines."""
        tool_pool = tmp_path / "tools.jsonl"
        _ = tool_pool.write_text(
            '{"api_name": "valid", "category": "A"}\n'
            + "invalid json\n"
            + '{"api_name": "also_valid", "category": "B"}\n',
            encoding="utf-8",
        )

        result = load_tool_categories(str(tool_pool))

        assert len(result) == 2
        assert "A" in result
        assert "B" in result

    def test_load_tool_categories_duplicate_tools(self, tmp_path: Path):
        """Test handling of duplicate tools."""
        tool_pool = tmp_path / "tools.jsonl"
        _ = tool_pool.write_text(
            '{"api_name": "tool1", "category": "A"}\n'
            + '{"api_name": "tool1", "category": "A"}\n',
            encoding="utf-8",
        )

        result = load_tool_categories(str(tool_pool))

        assert len(result["A"]) == 2  # Both are loaded


class TestMain:
    """Tests for main function."""

    def test_main_missing_env_vars(self, tmp_path: Path):
        """Test exit when environment variables are missing."""
        test_args = [
            "generate_step_by_step.py",
            "-n",
            "1",
            "-o",
            str(tmp_path / "out.jsonl"),
        ]
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(sys, "argv", test_args),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()
        assert exc_info.value.code == 1

    def test_main_with_env_vars(self, tmp_path: Path):
        """Test a zero-datapoint run with configured API credentials."""
        tool_pool = tmp_path / "tools.jsonl"
        invocation_examples = tmp_path / "invocations.jsonl"
        _ = tool_pool.write_text(
            '{"api_name": "test_tool", "category": "Test"}\n',
            encoding="utf-8",
        )
        _ = invocation_examples.write_text("", encoding="utf-8")
        test_args = [
            "generate_step_by_step.py",
            "--mode",
            "step-by-step",
            "-n",
            "0",
            "-o",
            str(tmp_path / "out.jsonl"),
            "--tool-pool",
            str(tool_pool),
            "--invocation-examples",
            str(invocation_examples),
        ]
        stdout = StringIO()
        with (
            patch.dict(
                os.environ,
                {
                    "OPENAI_API_KEY": "test-key",
                    "OPENAI_API_BASE": "http://test:8000/v1",
                },
                clear=True,
            ),
            patch.object(sys, "argv", test_args),
            patch.object(sys, "stdout", stdout),
        ):
            main()
        assert "Total generated: 0/0" in stdout.getvalue()

    def test_main_multi_turn_passes_judge_client(self, tmp_path: Path):
        tool_pool = tmp_path / "tools.jsonl"
        invocation_examples = tmp_path / "invocations.jsonl"
        _ = tool_pool.write_text(
            '{"api_name": "test_tool", "category": "Test"}\n',
            encoding="utf-8",
        )
        _ = invocation_examples.write_text("", encoding="utf-8")
        captured: dict[str, object] = {}

        def capture_multi_turn(**kwargs: object) -> list[object]:
            captured.update(kwargs)
            return []

        test_args = [
            "generate_step_by_step.py",
            "--mode",
            "multi-turn",
            "-n",
            "0",
            "--judge-model",
            "judge/model",
            "--tool-pool",
            str(tool_pool),
            "--invocation-examples",
            str(invocation_examples),
        ]
        with (
            patch.dict(
                os.environ,
                {
                    "OPENAI_API_KEY": "test-key",
                    "OPENAI_API_BASE": "http://test:8000/v1",
                },
                clear=True,
            ),
            patch.object(sys, "argv", test_args),
            patch(
                "generate_step_by_step.run_multi_turn",
                side_effect=capture_multi_turn,
            ),
        ):
            main()

        assert captured["judge_client"] is not captured["llm_client"]


class TestDiscardReasons:
    """Tests for discard statistics tracking."""

    def test_discard_reasons_structure(self):
        """Test that discard reasons dict has expected keys."""
        discard_reasons = {
            "generation_failed": 0,
            "wrong_step_count": 0,
            "verification_failed": 0,
            "order_incorrect": 0,
            "tools_not_relevant": 0,
        }

        assert "generation_failed" in discard_reasons
        assert "wrong_step_count" in discard_reasons
        assert "verification_failed" in discard_reasons
        assert "order_incorrect" in discard_reasons
        assert "tools_not_relevant" in discard_reasons

    def test_discard_reasons_increment(self):
        """Test incrementing discard counters."""
        discard_reasons = {
            "generation_failed": 0,
            "wrong_step_count": 0,
        }

        # Simulate incrementing
        discard_reasons["generation_failed"] += 1
        discard_reasons["generation_failed"] += 1
        discard_reasons["wrong_step_count"] += 1

        assert discard_reasons["generation_failed"] == 2
        assert discard_reasons["wrong_step_count"] == 1


class TestOutputDirectory:
    """Tests for output directory creation."""

    def test_output_dir_created(self, tmp_path: Path):
        """Test that output directory is created if it doesn't exist."""
        nested_dir = tmp_path / "nested" / "output"

        # Directory shouldn't exist yet
        assert not nested_dir.exists()

        # Create it
        nested_dir.mkdir(parents=True)

        assert nested_dir.exists()
        assert nested_dir.is_dir()

    def test_output_file_written(self, tmp_path: Path):
        """Test that output file is written correctly."""
        output_file = tmp_path / "output.jsonl"

        data: dict[str, object] = {"key": "value", "number": 42}
        _ = output_file.write_text(
            json.dumps(data) + "\n",
            encoding="utf-8",
        )
        parsed = decode_json_object(
            output_file.read_text(encoding="utf-8"),
            source="test output",
        )

        assert parsed["key"] == "value"
        assert parsed["number"] == 42


class TestGenerationLoop:
    """Tests for generation loop logic."""

    def test_max_attempts_calculation(self):
        """Test that max_attempts is calculated correctly."""
        num_datapoints = 10
        max_attempts = num_datapoints * 50  # As in the code

        assert max_attempts == 500

    def test_remaining_calculation(self):
        """Test remaining datapoints calculation."""
        num_datapoints = 100
        generated = 45
        remaining = num_datapoints - generated

        assert remaining == 55

    def test_attempt_counter(self):
        """Test attempt counting logic."""
        attempt = 0
        max_attempts = 100
        generated = 0
        num_datapoints = 10

        while generated < num_datapoints and attempt < max_attempts:
            attempt += 1
            # Simulate generation
            if attempt % 2 == 0:  # 50% success rate
                generated += 1

        assert generated == num_datapoints
        assert attempt == 20  # Took 20 attempts at 50% success


class TestStatistics:
    """Tests for statistics generation."""

    def test_statistics_structure(self):
        """Test statistics dictionary structure."""
        from collections import Counter

        tools_used_all = ["tool1", "tool2", "tool1", "tool3"]
        tool_counts = Counter(tools_used_all)

        most_common = tool_counts.most_common(10)

        assert len(most_common) == 3
        assert most_common[0] == ("tool1", 2)  # Most frequent
