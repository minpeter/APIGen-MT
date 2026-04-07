"""
collect_tools.py — Collect and unify tool definitions from all datasets used
by the Magnet paper (arxiv 2503.07826).

Sources
-------
1. **StableToolBench** (HuggingFace: stabletoolbench/ToolEnv2404)
   Download instructions:
       huggingface-cli download stabletoolbench/ToolEnv2404 --repo-type dataset --local-dir ./data/ToolEnv2404
   Expected local path: data/ToolEnv2404/tools/

2. **BFCL-v3 multi-turn func docs** (HuggingFace: gorilla-llm/Berkeley-Function-Calling-Leaderboard)
   Download instructions:
       huggingface-cli download gorilla-llm/Berkeley-Function-Calling-Leaderboard --repo-type dataset --local-dir ./data/BFCL_v3
   Expected local paths:
       data/BFCL_v3/data/multi_turn_func_doc/
       data/BFCL_v3/data/BFCL_v3_multi_turn_*.json   (used to discover classes)

Output
------
A single JSONL file (one tool definition per line) written to
``output/tool_pool.jsonl`` by default.

Usage
-----
    python collect_tools.py \
        --stabletoolbench-tools data/ToolEnv2404/tools \
        --bfcl-func-doc       data/BFCL_v3/data/multi_turn_func_doc \
        --bfcl-data-dir       data/BFCL_v3/data \
        --output              output/tool_pool.jsonl
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Allow running from the src/ directory or the project root.
sys.path.insert(0, str(Path(__file__).parent))

from parse_stabletoolbench import parse_stable_toolbench
from parse_bfcl import parse_bfcl_func_doc, discover_bfcl_classes, parse_bfcl_jsonl
from tool_definition import ToolDefinition

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def deduplicate(definitions: list[ToolDefinition]) -> list[ToolDefinition]:
    """
    Remove duplicate API definitions.

    Two definitions are considered duplicates if they share the same
    (tool_name, api_name) pair (case-insensitive).
    """
    seen: set[tuple[str, str]] = set()
    unique: list[ToolDefinition] = []
    for defn in definitions:
        key = (defn.tool_name.lower(), defn.api_name.lower())
        if key in seen:
            continue
        seen.add(key)
        unique.append(defn)
    return unique


def collect(
    stabletoolbench_tools: Path | None,
    bfcl_func_doc: Path | None,
    bfcl_data_dir: Path | None,
    *,
    require_parameters: bool = True,
) -> list[ToolDefinition]:
    """
    Collect tool definitions from all configured sources.

    Args:
        stabletoolbench_tools: Path to the StableToolBench ``tools/`` directory.
        bfcl_func_doc: Path to the BFCL ``multi_turn_func_doc/`` directory.
        bfcl_data_dir: Path to the BFCL ``data/`` directory (used to discover
                       which classes appear in the test set).
        require_parameters: Whether to filter out parameter-less APIs.

    Returns:
        Deduplicated list of :class:`ToolDefinition`.
    """
    all_definitions: list[ToolDefinition] = []

    # --- StableToolBench --------------------------------------------------
    if stabletoolbench_tools is not None:
        logger.info("Parsing StableToolBench tools from: %s", stabletoolbench_tools)
        stb_defs = parse_stable_toolbench(
            stabletoolbench_tools,
            require_parameters=require_parameters,
        )
        logger.info("  → %d API definitions from StableToolBench", len(stb_defs))
        all_definitions.extend(stb_defs)
    else:
        logger.info("StableToolBench path not provided — skipping.")

    # --- BFCL-v3 ----------------------------------------------------------
    if bfcl_func_doc is not None:
        # Optionally filter to only classes referenced in the test files.
        class_filter: list[str] | None = None
        if bfcl_data_dir is not None:
            discovered = discover_bfcl_classes(bfcl_data_dir)
            if discovered:
                class_filter = sorted(discovered)
                logger.info(
                    "  Using %d discovered BFCL classes as filter: %s",
                    len(class_filter),
                    class_filter,
                )

        logger.info("Parsing BFCL-v3 func-doc from: %s", bfcl_func_doc)
        bfcl_defs = parse_bfcl_func_doc(
            bfcl_func_doc,
            class_names=class_filter,
            require_parameters=require_parameters,
        )
        logger.info("  → %d function definitions from BFCL-v3", len(bfcl_defs))
        all_definitions.extend(bfcl_defs)
    else:
        logger.info("BFCL func-doc path not provided — skipping.")

    # --- BFCL Generic (JSONL) ----------------------------------------------
    if bfcl_data_dir is not None:
        logger.info("Parsing BFCL-v3 Generic tools from: %s", bfcl_data_dir)
        bfcl_generic_defs = parse_bfcl_jsonl(
            bfcl_data_dir,
            require_parameters=require_parameters,
        )
        logger.info("  → %d API definitions from BFCL Generic", len(bfcl_generic_defs))
        all_definitions.extend(bfcl_generic_defs)

    # --- Deduplication ----------------------------------------------------
    before = len(all_definitions)
    all_definitions = deduplicate(all_definitions)
    logger.info(
        "Deduplication: %d → %d definitions (%d removed)",
        before,
        len(all_definitions),
        before - len(all_definitions),
    )

    return all_definitions


def save_jsonl(definitions: list[ToolDefinition], output_path: Path) -> None:
    """Serialize definitions to a JSON-lines file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for defn in definitions:
            fh.write(json.dumps(defn.to_dict(), ensure_ascii=False) + "\n")
    logger.info("Wrote %d definitions to %s", len(definitions), output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect tool definitions à la Magnet (arxiv 2503.07826)."
    )
    parser.add_argument(
        "--stabletoolbench-tools",
        type=Path,
        default=None,
        help=(
            "Path to the StableToolBench 'tools/' directory "
            "(e.g. data/ToolEnv2404/tools)."
        ),
    )
    parser.add_argument(
        "--bfcl-func-doc",
        type=Path,
        default=None,
        help=(
            "Path to the BFCL 'multi_turn_func_doc/' directory "
            "(e.g. data/BFCL_v3/data/multi_turn_func_doc)."
        ),
    )
    parser.add_argument(
        "--bfcl-data-dir",
        type=Path,
        default=None,
        help=(
            "Path to the BFCL 'data/' directory used to auto-discover the "
            "classes referenced in the multi-turn test files. Optional."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/tool_pool.jsonl"),
        help="Output JSONL file path (default: output/tool_pool.jsonl).",
    )
    parser.add_argument(
        "--no-require-parameters",
        action="store_true",
        help="Include APIs with no parameters (disabled by default, matching Magnet's filter).",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print per-category statistics after collection.",
    )
    args = parser.parse_args()

    if args.stabletoolbench_tools is None and args.bfcl_func_doc is None:
        parser.error(
            "At least one of --stabletoolbench-tools or --bfcl-func-doc must be provided."
        )

    definitions = collect(
        stabletoolbench_tools=args.stabletoolbench_tools,
        bfcl_func_doc=args.bfcl_func_doc,
        bfcl_data_dir=args.bfcl_data_dir,
        require_parameters=not args.no_require_parameters,
    )

    save_jsonl(definitions, args.output)

    if args.stats:
        from collections import Counter
        counts = Counter(d.category for d in definitions)
        print("\nPer-category counts:")
        for cat, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"  {cat:<40} {cnt}")
        print(f"\nTotal: {len(definitions)} definitions across {len(counts)} categories")


if __name__ == "__main__":
    main()
