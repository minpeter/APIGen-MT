"""Compatibility command-line entry point for single-datapoint generation."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from llm_local_openai_client import LocalOpenAILLMClient
from tool_manager import ToolManager


def run_cli() -> None:
    """Run the original single-datapoint command-line entry point."""
    _ = load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE")

    if not api_key or not api_base:
        print("ERROR: OPENAI_API_KEY or OPENAI_API_BASE not set")
        sys.exit(1)

    llm_client = LocalOpenAILLMClient(
        url=api_base,
        api_key=api_key,
        api_model="z-ai/glm-5.1",
        hf_tokenizer_id=None
    )

    tool_pool_path = str(Path("~/data/APIGen-MT/magnet_tool_extraction/bfcl_v3_tools_with_outputs.jsonl").expanduser())
    invocation_examples_path = str(Path("~/data/APIGen-MT/magnet_tool_extraction/bfcl_v3_invocation_examples.jsonl").expanduser())
    tool_manager = ToolManager(
        llm=llm_client,
        tool_pool_path=tool_pool_path,
        invocation_examples_path=invocation_examples_path
    )

    from apigen_step_by_step import StepByStepGenerator

    generator = StepByStepGenerator(
        llm_client=llm_client,
        tool_manager=tool_manager
    )

    print("Generating test datapoint...")
    datapoint = generator.generate_datapoint(focus_category="Communication")

    if datapoint:
        print("\n" + "=" * 60)
        print("GENERATED DATAPOINT:")
        print("=" * 60)
        print(datapoint.model_dump_json(indent=2))
    else:
        print("\nFailed to generate datapoint")


if __name__ == "__main__":
    run_cli()
