# APIGen-MT Simulation (Python Implementation)

This project provides a Python implementation of the core concepts in **APIGen-MT**, described in "[APIGen-MT: Agentic Pipeline for Multi-Turn Data Generation via Simulated Agent-Human Interplay](https://arxiv.org/abs/2504.03601v4)" (arXiv:2504.03601).

The goal is to generate synthetic multi-turn conversational data for training AI agents capable of tool use, following the two-phase approach outlined in the paper:
1.  **Phase 1:** Generation and validation of a task "blueprint" containing a user query (`q`), a ground-truth sequence of tool calls (`a_gt`), and an expected outcome description (`o_gt`).
2.  **Phase 2:** Simulation of a multi-turn interaction between a user and an AI agent based on the validated blueprint, resulting in a conversational trajectory including dialogue and tool usage.

The primary CLI uses an OpenAI-compatible API (via the `openai`-compatible HTTP client) and leverages **Function Calling** for the agent's decision-making process in Phase 2. The default endpoint and model are defined once in `src/runtime_config.py`.

## Implemented Features

* **Two-Phase Pipeline:** Implements the core structure of generating blueprints first, followed by trajectory simulation.
* **Phase 1: Blueprint Generation & Validation:**
    * LLM-based generation of blueprint candidates (`q`, `a_gt`, `o_gt`).
    * Manual parsing of LLM response to extract JSON blueprint.
    * Pydantic models (`Blueprint`, `ToolCallInternal`, `GetWeatherArgs`, `GetNewsArgs`) for blueprint structure validation and tool argument validation.
    * Basic execution check (verifies tool name existence and argument structure against Pydantic models).
    * Simplified LLM-based review step to assess blueprint coherence (`pass`/`fail`).
    * **Blueprint Deduplication:** Uses SHA-256 hashing of canonical blueprint representation to detect and skip generating trajectories for duplicate blueprints.
* **Phase 2: Trajectory Simulation (using Function Calling):**
    * Multi-turn simulation loop managing interactions between simulated user and agent.
    * Basic LLM-based user simulation (first turn uses blueprint `q`, subsequent turns use LLM).
    * LLM-based agent simulation using **Function Calling** (`tools` parameter, `tool_choice="auto"`).
    * Execution of predefined Python tool functions (`get_weather`, `get_news`) which currently return **fixed fake data**.
    * Parsing and validation of tool arguments requested by the LLM.
    * Handling of `tool_calls` response from the LLM.
    * Sending tool execution results back to the LLM (`role="tool"` messages).
    * Generating final text responses from the agent after tool use.
    * **Message Sanitization:** Cleans assistant messages before adding them to the API call history to remove non-standard fields (like `refusal`) that can cause errors.
    * **Trajectory Validation:** Validates the completed trajectory by comparing the sequence of *successfully executed* tool calls against the ground truth sequence (`a_gt`) from the blueprint.
    * Saving successful, unique data points (blueprint + trajectory) to a `.jsonl` file.
    * Removes `null` values from the final trajectory data before saving.
* **Configuration:** Uses `OPENAI_API_KEY`, optional `OPENAI_API_BASE`, and the shared default model `minimax/minimax-m2.7`. `--model` and `--judge-model` override the model for a run.

## How it Works (Simplified Flow)

1.  **Initialize:** Define tools (Python functions and API definitions), set up API client. Initialize an empty set to track seen blueprint IDs.
2.  **Loop to Generate Data:** Repeat until the desired number of unique data points (`num_data_to_generate`) is reached:
    * **Inner Attempt Loop:** Try up to `max_attempts_per_data` times to generate one valid data point.
        * **Generate Blueprint (1-1):** Call LLM (`MODEL_ID`) to propose a blueprint JSON string based on tool descriptions.
        * **Parse & Validate Blueprint (1-1b):** Parse the string and validate its structure using the `Blueprint` Pydantic model.
        * **Check Execution (1-2):** Verify if tools in `a_gt` exist and arguments match expected structure (using Pydantic arg models).
        * **Review Blueprint (1-3):** Call LLM (`MODEL_ID`) to check if the blueprint is coherent and logically sound.
        * **Check for Duplicates:** If all checks pass, generate a unique ID (hash) for the blueprint. If this ID has been seen before, skip to the next attempt.
        * **(If Blueprint is Valid & Unique) Start Trajectory Simulation (Phase 2):**
            * Initialize `history`, `trajectory`, `executed_actions`. Add system message to `history`.
            * **Simulate First User Turn (2-1):** Get initial user utterance (currently uses `blueprint.q`). Add to history/trajectory.
            * **Turn Loop:**
                * **Agent Turn (2-2):** Call LLM (`FC_MODEL_ID`) with `history` and `tools`, using `tool_choice="auto"`. Sanitize response and add to history/trajectory.
                * **Tool Handling (2-3, 2-4):** If `tool_calls` received:
                    * Execute tools using `available_tools_mapping`, get results (fake data).
                    * Record successfully executed calls in `executed_actions`.
                    * Add `role="tool"` messages to history/trajectory.
                    * Call LLM (`FC_MODEL_ID` with `tool_choice="none"`) again with updated history to get the final text response based on tool results. Sanitize response and add to history/trajectory.
                * **(If No Tool Calls) Text Response:** Use the agent's direct text response.
                * **Next User Turn (2-5):** Determine conversation context, call LLM (`MODEL_ID`) to simulate user's next utterance. Add to history/trajectory.
                * **Check End Conditions:** Break loop if user indicates ending, ground truth actions completed + agent responded, or max turns reached.
            * **Validate Trajectory (2-6):** Compare `executed_actions` list against `blueprint.a_gt`.
            * **Success:** If validation passes, add the blueprint ID to `seen_blueprint_ids`, increment `generated_count`, save the (blueprint, cleaned_trajectory) pair to `final_dataset`, and break the inner attempt loop (move to generate next data point).
            * **Failure:** If validation fails, continue to the next attempt in the inner loop.
3.  **Save Dataset:** After the outer loop finishes (or hits max attempts), save the collected `final_dataset` to a `.jsonl` file, cleaning out `null` values from trajectory messages.

## Setup and Requirements

1.  **Python:** Python 3.13 (the version declared by `pyproject.toml`).
2.  **Libraries:** Install required libraries. Pydantic v2+ and openai v1.17.0+ are recommended.
    ```bash
    uv sync --all-extras
    ```
3.  **API credentials:** Set the required key and, optionally, an endpoint override:
    ```bash
    export OPENAI_API_KEY="your-api-key"
    # Defaults to https://openrouter.ai/api/v1
    export OPENAI_API_BASE="https://openrouter.ai/api/v1"
    ```

## How to Run

Install the project, test dependencies, and ML extras:

```bash
uv sync --all-extras
```

Set `OPENAI_API_KEY` and, when needed, `OPENAI_API_BASE`, then inspect the CLI:

```bash
uv run python main.py --help
```

The default model is `minimax/minimax-m2.7`. `OPENAI_API_BASE` is optional and defaults to `https://openrouter.ai/api/v1`; `OPENAI_API_KEY` remains required only when generation starts. `uv run python main.py --help` and `uv run pytest tests/unit/test_runtime_config.py` are no-network smoke checks.

Generate step-by-step or multi-turn data:

```bash
uv run python main.py --mode step-by-step --num-datapoints 10 --num-actions 2
uv run python main.py --mode multi-turn --num-datapoints 10 --num-turns 3
```

Deterministically replay a generated JSON or JSONL trajectory without an LLM call:

```bash
uv run python main.py \
  --verify-trajectory data/generated/example.jsonl \
  --verification-output verification.json
```

The verification command exits `0` for reproducible trajectories, `1` for output/state mismatches, and `2` when deterministic verification is unavailable or the input is invalid.

## CI and typing baseline

CI installs the locked project with `uv sync --all-extras --frozen`, runs the full pytest suite, compiles `src/`, `tools/`, and `tests/`, and runs `basedpyright` on `src/`. The maintained `src/` boundary currently has a clean static-type baseline. A repository-wide baseline measurement on `main` found 180 errors and 3,404 warnings, concentrated in generated tool implementations and legacy extraction/generator scripts. Those surfaces are intentionally outside this gate with an explicit rollout boundary; no individual diagnostics are blanket-disabled. They should be brought into the gate incrementally with ownership and tests.

## Implemented Validation

For tools with local Python implementations, final acceptance now:

1. Restores the recorded initial API state.
2. Replays resolved tool calls.
3. Compares structured outputs and recorded pre/post state.
4. Rejects replay errors or invalid recorded state verdicts.
5. Restores the caller's live state even when replay fails.
6. Stores verification results for both step-by-step and multi-turn datapoints.

Tools without a local implementation are marked `unavailable`; they are never reported as deterministically verified.

## Remaining Differences from the Paper

The official APIGen-MT generation pipeline is not publicly released, so this repository does not claim source-level reproduction. Material differences remain:

* Context sampling uses local configuration/persona seeds rather than the paper's API dependency-graph random walks.
* One configurable judge is available; the paper's independent reviewer committee and majority policy are not reproduced.
* Reverse task recombination and Best-of-N user simulation are not implemented.
* Deterministic replay proves artifact integrity. It does not replace domain-specific semantic assertions over independent `o_gt` targets.
* LLM-simulated tools remain non-authoritative until a local executable implementation or task-specific verifier exists.

These omissions are deliberate. Follow-up stateful-agent work consistently favors executable outcome checks before adding more generation or reviewer complexity:

* [tau-bench](https://arxiv.org/abs/2406.12045)
* [AppWorld](https://arxiv.org/abs/2407.18901)
* [ToolSandbox](https://arxiv.org/abs/2408.04682)
* [tau2-bench](https://arxiv.org/abs/2506.07982)
* [ToolACE-MT](https://arxiv.org/abs/2508.12685)
* [EigenData](https://arxiv.org/abs/2601.22607)
* [ToolMind](https://arxiv.org/abs/2511.15718)

## Reference

This project is based on the concepts presented in the APIGen-MT paper:

* **Paper (arXiv):** [https://arxiv.org/abs/2504.03601v4](https://arxiv.org/abs/2504.03601v4)
* **Paper (HTML):** [https://arxiv.org/html/2504.03601v4](https://arxiv.org/html/2504.03601v4)
* **Project Website:** [https://apigen-mt.github.io/](https://apigen-mt.github.io/)

**Citation:**

If you use concepts or code derived from this simulation or the original paper, please cite:

```bibtex
@article{prabhakar2025apigenmt,
  title={APIGen-MT: Agentic Pipeline for Multi-Turn Data Generation via Simulated Agent-Human Interplay},
  author={Prabhakar, Akshara and Liu, Zuxin and Zhu, Ming and Zhang, Jianguo and Awalgaonkar, Tulika and Wang, Shiyu and Liu, Zhiwei and Chen, Haolin and Hoang, Thai and Niebles, Juan Carlos and Heinecke, Shelby and Yao, Weiran and Wang, Huan and Savarese, Silvio and Xiong, Caiming},
  journal={arXiv preprint arXiv:2504.03601},
  year={2025}
}
```

## License

The original APIGen-MT paper and associated resources are under the [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/). This implementation attempts to simulate the described concepts.
