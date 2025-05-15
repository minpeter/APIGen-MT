# APIGen-MT Simulation (Python Implementation)

> [!NOTE]
> WIP: I probably won't be able to finish this task :/

This project provides a Python implementation simulating the core concepts of the **APIGen-MT** framework described in the paper "[APIGen-MT: Agentic PIpeline for Multi-Turn Data Generation via Simulated Agent-Human Interplay](https://arxiv.org/abs/2504.03601v2)" (arXiv:2504.03601).

The goal is to generate synthetic multi-turn conversational data for training AI agents capable of tool use, following the two-phase approach outlined in the paper:
1.  **Phase 1:** Generation and validation of a task "blueprint" containing a user query (`q`), a ground-truth sequence of tool calls (`a_gt`), and an expected outcome description (`o_gt`).
2.  **Phase 2:** Simulation of a multi-turn interaction between a user and an AI agent based on the validated blueprint, resulting in a conversational trajectory including dialogue and tool usage.

This implementation utilizes Large Language Models (LLMs) via the Friendli.ai API (using the `openai` library) and leverages **Function Calling** for the agent's decision-making process in Phase 2.

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
* **Configuration:** Allows specifying different LLM models for standard generation (`MODEL_ID`) and function calling steps (`FC_MODEL_ID`).

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

1.  **Python:** Python 3.8+ recommended.
2.  **Libraries:** Install required libraries. Pydantic v2+ and openai v1.17.0+ are recommended.
    ```bash
    pip install openai "pydantic>=2.0"
    ```
    (Consider creating a `requirements.txt` file).
3.  **API Key:** Set your Friendli.ai API token as an environment variable:
    ```bash
    export FRIENDLI_TOKEN="your_friendli_api_token_here"
    ```

## How to Run

1.  Save the code as a Python file (e.g., `apigen_mt_simulation.py`).
2.  Set the `FRIENDLI_TOKEN` environment variable.
3.  Run the script from your terminal:
    ```bash
    python apigen_mt_simulation.py
    ```
4.  The script will print progress and save the generated data to a `.jsonl` file upon completion.

## Limitations and Differences from APIGen-MT Paper

This implementation is a simplified simulation of the APIGen-MT framework and has several limitations compared to the methodology described in the paper:

* **Simplified Context:** Does not use the paper's sophisticated context preparation (API graph, policy/domain data/persona samplers). Blueprints are generated based only on basic tool descriptions.
* **Simplified Validation (Phase 1):**
    * Execution checks are basic (tool existence, argument structure via Pydantic) and do not involve simulating a real environment state.
    * **Policy compliance checks are not implemented.**
    * The review committee is simulated with a single LLM call, lacking multi-judge voting.
    * **The agentic feedback/refinement loop based on validation/review failures is not implemented** (only basic retries).
* **Missing Techniques:** Does not implement Reverse Task Recombination for complex task generation.
* **Fake Environment (Phase 2):** Tool functions return fixed fake data, not interacting with a real, stateful environment.
* **Simplified Trajectory Validation (Phase 2):** Validation only compares the sequence of executed tool calls (`executed_actions`) against the ground truth (`a_gt`). It does not perform state-based or output-based (`o_gt`) validation described in the paper.
* **Basic User Simulation (Phase 2):** User simulation lacks persona integration and advanced stabilization techniques (Best-of-N sampling, self-critique).
* **No Handling of Failed Trajectories:** Failed trajectories are simply discarded; they are not used for potential contrastive learning.

In essence, this code serves as a functional proof-of-concept for the core two-phase structure and function calling integration, but lacks the advanced components required to generate the high-fidelity, diverse, and robustly validated data described in the APIGen-MT paper.

## Future Work (TODOs)

* **Implement Advanced Context Samplers:** Integrate policy, domain data, persona samplers, and potentially API graph logic for richer blueprint generation (Phase 1).
* **Implement Policy Compliance Checks:** Add mechanisms to verify blueprint actions and trajectory actions against predefined rules (Phase 1 & 2).
* **Enhance Blueprint Review:** Implement multi-LLM review committee with majority voting (Phase 1).
* **Implement Feedback Loop:** Add logic for the LLM generator to reflect on failure reasons and refine blueprints (Phase 1).
* **Implement Reverse Task Recombination:** Allow building complex blueprints from simpler validated ones (Phase 1).
* **Integrate Real Environment:** Replace fake tool functions with calls to a real executable environment or a more sophisticated simulator (Phase 2).
* **Implement Advanced Trajectory Validation:** Add state-based checks and comparison against `o_gt` (Phase 2).
* **Enhance User Simulation:** Integrate personas and stabilization techniques (BoN, self-critique) (Phase 2).
* **Explore Failed Trajectory Use:** Investigate methods to leverage failed trajectories for training (e.g., contrastive learning).

## Reference

This project is based on the concepts presented in the APIGen-MT paper:

* **Paper (arXiv):** [https://arxiv.org/abs/2504.03601v2](https://arxiv.org/abs/2504.03601v2)
* **Paper (HTML):** [https://arxiv.org/html/2504.03601v2](https://arxiv.org/html/2504.03601v2)
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
