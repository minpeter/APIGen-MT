"""Generate step-by-step or multi-turn tool-calling datapoints.

Use ``--checkpoint`` to save progress after each turn and resume an interrupted
multi-turn generation. Use ``--verify-trajectory`` to deterministically replay a
saved JSON or JSONL trajectory without starting an LLM client.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from dotenv import load_dotenv


@runtime_checkable
class LineBufferedStream(Protocol):
    def reconfigure(self, *, line_buffering: bool) -> None: ...


if isinstance(sys.stdout, LineBufferedStream):
    sys.stdout.reconfigure(line_buffering=True)
if isinstance(sys.stderr, LineBufferedStream):
    sys.stderr.reconfigure(line_buffering=True)

_ = load_dotenv()
_ = sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_ = sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

from apigen_multi_turn import MultiTurnDatapoint, MultiTurnGenerator
from apigen_step_by_step import StepByStepDatapoint, StepByStepGenerator
from llm_client import LocalOpenAILLMClient
from llm_request_helpers import decode_json_object
from runtime_config import DEFAULT_MODEL, RuntimeConfig
from step_by_step_models import TokenUsageStats
from step_by_step_protocols import LLM, StepByStepToolManager
from tool_manager import ToolManager
from trajectory_cli import verify_trajectory_file

type Checkpoint = dict[str, object]
type GenerationMode = Literal["multi-turn", "step-by-step"]
type ToolSchema = dict[str, object]
type ToolsByCategory = dict[str, list[ToolSchema]]


class Arguments(argparse.Namespace):
    """Typed command-line values populated by argparse."""

    mode: GenerationMode = "multi-turn"
    num_datapoints: int = 100
    num_turns: int = 10
    num_actions: int = 1
    output: str = "step_by_step_datapoints.jsonl"
    tool_pool: str = ""
    invocation_examples: str = ""
    category: str | None = None
    model: str = DEFAULT_MODEL
    judge_model: str | None = None
    judge_api_base: str | None = None
    judge_api_key: str | None = None
    verify_trajectory: str | None = None
    verification_output: str = "trajectory_verification.json"
    config_pool: bool = True
    checkpoint: str | None = None
    resume: bool = True


class CheckpointManager:
    """Manage checkpoint persistence for resumable generation."""

    def __init__(self, checkpoint_path: str):
        self.checkpoint_path: str = checkpoint_path
        self.state: Checkpoint = {}

    def load(self) -> Checkpoint:
        """Load checkpoint state if it exists and contains an object."""
        path = Path(self.checkpoint_path)
        if not self.checkpoint_path or not path.exists():
            return {}
        try:
            self.state = decode_json_object(
                path.read_text(encoding="utf-8"),
                source="checkpoint",
            )
        except (json.JSONDecodeError, OSError, TypeError):
            return {}
        return self.state

    def save(self, state: Checkpoint) -> None:
        """Save checkpoint state to disk."""
        if not self.checkpoint_path:
            return
        self.state = state
        try:
            _ = Path(self.checkpoint_path).write_text(
                json.dumps(state, default=str),
                encoding="utf-8",
            )
        except OSError as error:
            print(f"Warning: Failed to save checkpoint: {error}")

    def clear(self) -> None:
        """Remove the checkpoint file if present."""
        path = Path(self.checkpoint_path)
        if self.checkpoint_path and path.exists():
            path.unlink()


def parse_args() -> Arguments:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate datapoints using step-by-step blueprint generation."
    )
    _ = parser.add_argument(
        "--mode",
        default="multi-turn",
        choices=["multi-turn", "step-by-step"],
        help="Generation mode: multi-turn (default) or step-by-step (legacy)",
    )
    _ = parser.add_argument(
        "--num-datapoints",
        "-n",
        type=int,
        default=100,
        help="Number of datapoints to generate (default: 100)",
    )
    _ = parser.add_argument(
        "--num-turns",
        "-t",
        type=int,
        default=10,
        help="Number of user-assistant turns for multi-turn mode (default: 10)",
    )
    _ = parser.add_argument(
        "--num-actions",
        "-a",
        type=int,
        default=1,
        help=(
            "Target tool actions per datapoint (step-by-step) or per turn "
            "(multi-turn) (default: 1)"
        ),
    )
    _ = parser.add_argument(
        "--output",
        "-o",
        default="step_by_step_datapoints.jsonl",
        help="Output file path (default: step_by_step_datapoints.jsonl)",
    )
    _ = parser.add_argument(
        "--tool-pool",
        default=(
            "~/data/APIGen-MT/magnet_tool_extraction/bfcl_v3_tools_with_outputs.jsonl"
        ),
        help="Path to tool pool file",
    )
    _ = parser.add_argument(
        "--invocation-examples",
        default=(
            "~/data/APIGen-MT/magnet_tool_extraction/bfcl_v3_invocation_examples.jsonl"
        ),
        help="Path to invocation examples for Python tool implementations",
    )
    _ = parser.add_argument(
        "--category",
        default=None,
        help="Filter tools to a specific category",
    )
    _ = parser.add_argument(
        "--model",
        "-m",
        default=DEFAULT_MODEL,
        help=f"Model to use for generation (default: {DEFAULT_MODEL})",
    )
    _ = parser.add_argument(
        "--judge-model",
        default=None,
        help="Judge model; defaults to --model",
    )
    _ = parser.add_argument(
        "--judge-api-base",
        default=None,
        help="Judge API URL; defaults to OPENAI_API_BASE",
    )
    _ = parser.add_argument(
        "--judge-api-key",
        default=None,
        help="Judge API key; defaults to OPENAI_API_KEY",
    )
    _ = parser.add_argument(
        "--verify-trajectory",
        default=None,
        help="Verify a trajectory JSON or JSONL file without generating data",
    )
    _ = parser.add_argument(
        "--verification-output",
        default="trajectory_verification.json",
        help="Output path for deterministic trajectory verification results",
    )
    _ = parser.add_argument(
        "--config-pool",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use diverse initial API states (default: True). "
            "Use --no-config-pool to disable."
        ),
    )
    _ = parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint file for resume support",
    )
    _ = parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Resume an existing checkpoint (default: True). "
            "Use --no-resume to start fresh."
        ),
    )
    return parser.parse_args(namespace=Arguments())


def load_tool_categories(tool_pool_path: str) -> ToolsByCategory:
    """Load tools and group them by category."""
    tools_by_category: ToolsByCategory = {}
    with Path(tool_pool_path).open(encoding="utf-8") as handle:
        for line in handle:
            try:
                tool = decode_json_object(line.strip(), source="tool schema")
            except (json.JSONDecodeError, TypeError):
                continue
            raw_category = tool.get("category", "Unknown")
            category = raw_category if isinstance(raw_category, str) else "Unknown"
            tools_by_category.setdefault(category, []).append(tool)
    return tools_by_category


def _timestamp() -> str:
    """Return a local naive timestamp matching the historical output shape."""
    return datetime.now(UTC).astimezone().replace(tzinfo=None).isoformat()


def _append_datapoint(path: str, payload: dict[str, object]) -> None:
    with Path(path).open("a", encoding="utf-8") as handle:
        _ = handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run_step_by_step(
    args: Arguments,
    llm_client: LLM,
    tool_manager: StepByStepToolManager,
    categories: list[str],
    output_path: str,
    judge_client: LLM | None = None,
) -> list[StepByStepDatapoint]:
    """Run legacy single-query generation."""
    generator = StepByStepGenerator(
        llm_client=llm_client,
        tool_manager=tool_manager,
        judge_client=judge_client,
        num_actions=args.num_actions,
    )
    datapoints: list[StepByStepDatapoint] = []
    attempt = 0

    while len(datapoints) < args.num_datapoints:
        remaining = args.num_datapoints - len(datapoints)
        print(f"\n{'=' * 70}")
        print(
            f"Generated: {len(datapoints)}/{args.num_datapoints} | "
            + f"Remaining: {remaining}"
        )
        print("=" * 70)
        focus_category = random.choice(categories)
        print(f"Focus category: {focus_category}")
        datapoint = generator.generate_datapoint(focus_category=focus_category)

        if datapoint:
            payload = datapoint.model_dump()
            payload["timestamp"] = _timestamp()
            payload["generation_attempt"] = attempt
            _append_datapoint(output_path, payload)
            datapoints.append(datapoint)
            print(
                "\n✓ Successfully generated and verified datapoint "
                + str(len(datapoints))
            )
            print(f" Query: {datapoint.trajectory.query}")
            print(f" Tools used: {datapoint.trajectory.tools_used}")
        else:
            print("\n✗ Failed to generate datapoint")
        attempt += 1

    return datapoints


def run_multi_turn(
    args: Arguments,
    llm_client: LLM,
    judge_client: LLM | None,
    tool_manager: StepByStepToolManager,
    categories: list[str],
    output_path: str,
    checkpoint_manager: CheckpointManager | None = None,
) -> list[MultiTurnDatapoint]:
    """Run multi-turn generation with optional checkpoint support."""
    generator = MultiTurnGenerator(
        llm_client=llm_client,
        tool_manager=tool_manager,
        num_turns=args.num_turns,
        num_actions=args.num_actions,
        judge_client=judge_client,
    )
    checkpoint_callback: Callable[[Checkpoint], None] | None = None
    if checkpoint_manager:
        checkpoint_callback = checkpoint_manager.save

    datapoints: list[MultiTurnDatapoint] = []
    while len(datapoints) < args.num_datapoints:
        remaining = args.num_datapoints - len(datapoints)
        print(f"\n{'=' * 70}")
        print(
            f"Generated: {len(datapoints)}/{args.num_datapoints} | "
            + f"Remaining: {remaining}"
        )
        print("=" * 70)

        if checkpoint_manager and args.resume:
            checkpoint = checkpoint_manager.load()
            if checkpoint.get("partial_conversation"):
                completed_turns = checkpoint.get("completed_turns", 0)
                print(f"\nFound checkpoint with {completed_turns} completed turns")
                raw_category = checkpoint.get("focus_category")
                focus_category = raw_category if isinstance(raw_category, str) else None
                datapoint = generator.continue_from_checkpoint(
                    checkpoint,
                    focus_category=focus_category,
                )
                if datapoint:
                    payload = datapoint.model_dump()
                    payload.update(
                        timestamp=_timestamp(),
                        resumed=True,
                        resumed_from_turn=completed_turns,
                    )
                    _append_datapoint(output_path, payload)
                    datapoints.append(datapoint)
                    checkpoint_manager.clear()
                    print(
                        "\n✓ Successfully resumed and completed datapoint "
                        + str(len(datapoints))
                    )
                    print(f" Task: {datapoint.conversation.overall_task[:80]}")
                    print(f" Turns: {len(datapoint.conversation.turns)}")
                    print(f" Tools: {datapoint.conversation.tools_used}")
                    continue
                print("\n✗ Failed to resume from checkpoint, starting fresh")
                checkpoint_manager.clear()

        focus_category = random.choice(categories)
        print(f"Focus category: {focus_category}")
        datapoint = generator.generate_multi_turn_datapoint(
            focus_category=focus_category,
            checkpoint_callback=checkpoint_callback,
        )
        if datapoint is None:
            print("\n✗ Failed to generate datapoint, retrying...")
            if checkpoint_manager:
                checkpoint_manager.clear()
            continue

        if checkpoint_manager:
            checkpoint_manager.clear()
        payload = datapoint.model_dump()
        payload["timestamp"] = _timestamp()
        _append_datapoint(output_path, payload)
        datapoints.append(datapoint)
        print(f"\n✓ Successfully generated datapoint {len(datapoints)}")
        print(f" Task: {datapoint.conversation.overall_task[:80]}")
        print(f" Turns: {len(datapoint.conversation.turns)}")
        print(f" Tools: {datapoint.conversation.tools_used}")

    return datapoints


def _print_summary(
    generated_count: int,
    target_count: int,
    output_path: str,
    tools_used: list[str],
    token_usages: list[TokenUsageStats],
) -> None:
    print(f"\n{'=' * 70}")
    print("GENERATION COMPLETE")
    print("=" * 70)
    print(f"Total generated: {generated_count}/{target_count}")
    print(f"Output file: {output_path}")
    if generated_count:
        print("\nTop 10 tools used:")
        for tool, count in Counter(tools_used).most_common(10):
            print(f"  {tool}: {count}")
        total_calls = sum(usage.total_llm_calls for usage in token_usages)
        total_tokens = sum(usage.total_tokens for usage in token_usages)
        print("\nToken Usage Statistics:")
        print(f"  Total LLM calls: {total_calls}")
        print(f"  Total tokens: {total_tokens:,}")
        print("  Average per datapoint:")
        print(f"    - LLM calls: {total_calls / generated_count:.1f}")
        print(f"    - Tokens: {total_tokens // generated_count:,}")
    print("=" * 70)


def _verify_only(args: Arguments, tool_pool: str, examples: str) -> None:
    """Execute deterministic verification and preserve its exit contract."""
    if args.verify_trajectory is None:
        raise ValueError("verification input is required")
    try:
        tool_manager = ToolManager(
            llm=None,
            tool_pool_path=tool_pool,
            invocation_examples_path=examples,
            use_config_pool=False,
        )
        verification, exit_code = verify_trajectory_file(
            str(Path(args.verify_trajectory).expanduser()),
            str(Path(args.verification_output).expanduser()),
            tool_manager,
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    print(json.dumps(verification, ensure_ascii=False))
    raise SystemExit(exit_code)


def main() -> None:
    args = parse_args()
    tool_pool_path = str(Path(args.tool_pool).expanduser())
    invocation_examples_path = str(Path(args.invocation_examples).expanduser())
    if args.verify_trajectory:
        _verify_only(args, tool_pool_path, invocation_examples_path)

    mode_label = "MULTI-TURN" if args.mode == "multi-turn" else "STEP-BY-STEP"
    print("=" * 70)
    print(f"{mode_label} DATAPOINT GENERATION")
    print("=" * 70)
    print(f"Target: {args.num_datapoints} datapoints")
    if args.mode == "multi-turn":
        print(f"Turns per conversation: {args.num_turns}")
    print(f"Output: {args.output}")
    print(f"Model: {args.model}")
    print("=" * 70)

    try:
        runtime_config = RuntimeConfig.from_environment(model=args.model)
    except ValueError as error:
        print(f"ERROR: {error}")
        raise SystemExit(1)

    llm_client = LocalOpenAILLMClient(
        url=runtime_config.api_base,
        api_key=runtime_config.api_key,
        api_model=runtime_config.model,
        hf_tokenizer_id=None,
    )
    print("\nLoading tools...")
    tools_by_category = load_tool_categories(tool_pool_path)
    if args.category:
        requested_tools = tools_by_category.get(args.category)
        if requested_tools is None:
            print(f"Error: Category '{args.category}' not found")
            print(f"Available categories: {list(tools_by_category)}")
            return
        tools_by_category = {args.category: requested_tools}

    total_tools = sum(len(tools) for tools in tools_by_category.values())
    print(
        f"Loaded {total_tools} tools across " + f"{len(tools_by_category)} categories"
    )
    for category, tools in sorted(tools_by_category.items()):
        print(f"  {category:30s}: {len(tools):3d} tools")

    tool_manager = ToolManager(
        llm=llm_client,
        tool_pool_path=tool_pool_path,
        invocation_examples_path=invocation_examples_path,
        use_config_pool=args.config_pool,
    )
    judge_client: LLM = llm_client
    if args.judge_model:
        judge_client = LocalOpenAILLMClient(
            url=args.judge_api_base or runtime_config.api_base,
            api_key=args.judge_api_key or runtime_config.api_key,
            api_model=args.judge_model,
            hf_tokenizer_id=None,
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    checkpoint_manager = CheckpointManager(args.checkpoint) if args.checkpoint else None
    if checkpoint_manager:
        print(f"Checkpoint file: {args.checkpoint}")
        print(f"Resume: {'enabled' if args.resume else 'disabled'}")

    categories = list(tools_by_category)
    if args.mode == "multi-turn":
        multi_datapoints = run_multi_turn(
            args=args,
            llm_client=llm_client,
            judge_client=judge_client,
            tool_manager=tool_manager,
            categories=categories,
            output_path=args.output,
            checkpoint_manager=checkpoint_manager,
        )
        generated_count = len(multi_datapoints)
        tools_used = [
            tool
            for datapoint in multi_datapoints
            for tool in datapoint.conversation.tools_used
        ]
        token_usages = [datapoint.token_usage for datapoint in multi_datapoints]
    else:
        step_datapoints = run_step_by_step(
            args,
            llm_client,
            tool_manager,
            categories,
            args.output,
            judge_client=judge_client,
        )
        generated_count = len(step_datapoints)
        tools_used = [
            tool
            for datapoint in step_datapoints
            for tool in datapoint.trajectory.tools_used
        ]
        token_usages = [datapoint.token_usage for datapoint in step_datapoints]

    _print_summary(
        generated_count,
        args.num_datapoints,
        args.output,
        tools_used,
        token_usages,
    )


if __name__ == "__main__":
    main()
