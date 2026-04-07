from enum import Enum
from pydantic import BaseModel, Field, ValidationError, field_validator
from typing import List, Dict, Any, Optional
import json
import re
import copy  # For deep copying objects
import os
from transformers import AutoTokenizer
from llm_client import LLMClient, LocalOpenAILLMClient
from tool_manager import ToolManager


# --- Pydantic Models for APIGen-MT Phase 1 (largely unchanged) ---


class ToolCallInternal(BaseModel):
    tool_name: str = Field(
        ..., description="Name of the tool to call"
    )
    arguments: Dict[str, Any] = Field(
        default_factory=dict, description="Arguments required for the tool call (dictionary format)"
    )


class AGTStep(BaseModel):
    tool_calls: List[ToolCallInternal] = Field(
        ..., description="Ground Truth: List of tool calls the agent should perform in this step"
    )


class Blueprint(BaseModel):
    q: str = Field(..., description="User's initial question or request")
    a_gt_steps: List[AGTStep] = Field(
        ..., description="Ground Truth Steps: List of tool call steps the agent should perform."
    )
    o_gt: str = Field(
        ..., description="Description of the expected final outcome (string)"
    )

    @field_validator('a_gt_steps')
    def check_a_gt_steps_not_empty(cls, v):
        if not v:
            raise ValueError('a_gt_steps must contain at least one step.')
        for step in v:
            if not step.tool_calls:
                raise ValueError('Each step in a_gt_steps must contain at least one tool call.')
        return v


class BlueprintCandidate(BaseModel):
    blueprint: Blueprint
    generation_reasoning: Optional[str] = None


class ValidationResult(BaseModel):
    is_valid_format: bool = False
    format_errors: Optional[List[str]] = None
    is_executable: bool = False
    executability_checks: List[Dict[str, Any]] = Field(default_factory=list)
    overall_validation_passed: bool = False


class QualityEnum(str, Enum):
    Excellent = "Excellent"
    Good = "Good"
    Fair = "Fair"
    Poor = "Poor"


class LLMReview(BaseModel):
    quality_assessment: QualityEnum = Field(
        ..., description="Quality assessment of the LLM review (one of Excellent, Good, Fair, Poor)"
    )
    feedback_summary: Optional[str] = None
    suggested_corrections: Optional[str] = None


class VerifiedBlueprint(BaseModel):
    blueprint: Blueprint
    validation_result: ValidationResult
    llm_review_history: List[LLMReview] = Field(default_factory=list)
    generation_attempts: int


# --- APIGen-MT Phase 1 Generator (Adapted) ---


class APIGenMTPhase1Generator:
    def __init__(self, llm_client: LLMClient, tool_manager: ToolManager):
        self.llm = llm_client
        self.tool_manager = tool_manager

    def _process_placeholders(
        self, arguments: Dict[str, Any], execution_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Processes placeholders in arguments using execution context from previous steps.
        Moved from the old ToolManager into this class.
        """
        processed_args = copy.deepcopy(arguments)
        for arg_name, arg_value in processed_args.items():
            if isinstance(arg_value, str):
                placeholders = re.findall(r"\{\{([^{}]+)\}\}", arg_value)
                for placeholder_full_key in placeholders:
                    keys = placeholder_full_key.split('.')
                    current_value = execution_context
                    found = True
                    for key_part in keys:
                        if isinstance(current_value, dict) and key_part in current_value:
                            current_value = current_value[key_part]
                        else:
                            # print(f"Warning: Placeholder key part '{key_part}' not found in context for '{placeholder_full_key}'")
                            found = False
                            break
                    if found:
                        placeholder_tag = "{{" + placeholder_full_key + "}}"
                        if arg_value == placeholder_tag:
                            processed_args[arg_name] = current_value
                        else:
                            processed_args[arg_name] = arg_value.replace(
                                placeholder_tag, str(current_value)
                            )
                    # else:
                    # print(f"Warning: Could not resolve placeholder '{placeholder_full_key}' for argument '{arg_name}'. Kept as is.")
        return processed_args

    def _generate_blueprint_candidate(
        self,
        q: str,
        previous_feedback: Optional[str] = None,
        previous_llm_reviews: Optional[List[LLMReview]] = None,
        previous_validation_result: Optional[ValidationResult] = None,
    ) -> Optional[BlueprintCandidate]:
        tool_schemas_list = self.tool_manager.get_tools_json_schema()
        tool_schema_json_str = json.dumps(tool_schemas_list, indent=2, ensure_ascii=False)

        llm_review_prompt = ""
        if previous_llm_reviews:
            llm_review_prompt += "There is a history of previous LLM reviews. Please refer to the quality assessments and feedback below to improve the blueprint.\n"
            for idx, review in enumerate(previous_llm_reviews, 1):
                llm_review_prompt += f"Review {idx}:\n"
                llm_review_prompt += f"- Quality Assessment: {review.quality_assessment}\n"
                if review.feedback_summary:
                    llm_review_prompt += f"- Feedback Summary: {review.feedback_summary}\n"
                if review.suggested_corrections:
                    llm_review_prompt += f"- Suggested Corrections: {review.suggested_corrections}\n"
                llm_review_prompt += "\n"

        validation_prompt = ""
        if previous_validation_result and not previous_validation_result.overall_validation_passed:
            validation_prompt += "The previous automatic validation result failed. Please refer to the validation results below to improve the blueprint.\n"
            if previous_validation_result.format_errors:
                validation_prompt += f"- Format Errors: {'; '.join(previous_validation_result.format_errors)}\n"
            if not previous_validation_result.is_executable:
                failed_checks = [
                    f"Step {chk['step_index']}-Tool {chk['tool_name']}: {chk['reason']}"
                    for chk in previous_validation_result.executability_checks
                    if not chk.get("can_execute")
                ]
                if failed_checks:
                    validation_prompt += f"- Reasons for Execution Failure: {'; '.join(failed_checks)}\n"
            validation_prompt += "\n"

        system_prompt = f"""You are a 'Task Blueprint Generator' for Phase 1 of the APIGen-MT pipeline. Your goal is to create a detailed blueprint representing a **realistic and verifiable multi-turn interaction scenario** between a user and an AI agent.

Based on the given tool descriptions, you must generate a **JSON object** containing the following components:

1. `q` (string): The user's initial question/request. It should be **specific and natural**, preferably representing a scenario that requires **multiple steps of interaction**, such as retrieving information, changing state, and re-confirming with the user.

2. `a_gt_steps` (list): A **Ground Truth tool call list** that the agent must call to **completely and in the correct order** resolve the user's request `q`. Each element must be in the format `{{"tool_name": "tool_name", "arguments": {{"arg_name": "value", ...}}}}` and must strictly match the available tool descriptions. It must include **at least one tool call**, and preferably scenarios requiring **two or more sequential tool calls**.

3. `o_gt` (string): A **natural description of the final summary or response message** that the agent should provide to the user, assuming all `a_gt` were executed successfully. It must reflect all results of `a_gt`.

**Reasoning Process (Optional but Recommended):**
- Before generating the final JSON, you may write your reasoning process (e.g., what scenario you envisioned, why you chose specific tools and sequences) inside `<think>...</think>` tags.

**Response Rules:**
- Your final response must contain a **valid JSON object** with the structure described above, inside a ```json ... ``` code block.
- Do **not** include any text other than the JSON object before or after the code block (except for an optional `<think>` block first).

{llm_review_prompt}{validation_prompt}

**Tool Descriptions:**
Available Tools and their Schemas:
{tool_schema_json_str}
"""

        user_prompt = f"User Query (q): {q}\n\n"
        if previous_feedback:
            user_prompt += f"Below is feedback on a previous attempt. Please incorporate this feedback to improve the blueprint:\n{previous_feedback}\n\n"
        user_prompt += "Using the guidelines and available tools above, generate the Blueprint JSON for this request."

        try:
            parsed_json, reasoning = self.llm.json_output(
                prompt=user_prompt,
                system_prompt=system_prompt,  # This detailed system prompt will be used by LLMClient
                schema=Blueprint,
                reasoning=True,
            )

            if parsed_json:
                blueprint_obj = Blueprint(**parsed_json)
                print(f"Debug: Generated Blueprint JSON: {json.dumps(parsed_json, indent=2, ensure_ascii=False)}")
                return BlueprintCandidate(blueprint=blueprint_obj, generation_reasoning=reasoning)
            else:
                print(f"Error: LLM did not return valid JSON for q: {q}")
                return None
        except ValidationError as e:
            print(f"Pydantic validation error for blueprint from LLM for q: {q}: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error during blueprint generation for q: {q}: {e}")
            return None

    def _validate_blueprint_format_and_executability(
        self, blueprint: Blueprint
    ) -> ValidationResult:
        format_errors: List[str] = []  # Should be caught by Pydantic mostly
        executability_checks: List[Dict[str, Any]] = []
        is_overall_executable = True
        current_execution_context: Dict[str, Any] = {}

        for step_idx, agt_step in enumerate(blueprint.a_gt_steps):
            step_executable = True
            step_results_for_context: Dict[str, Any] = {}

            for tool_idx, tool_call in enumerate(agt_step.tool_calls):
                check_result = {
                    "step_index": step_idx,
                    "tool_index_in_step": tool_idx,
                    "tool_name": tool_call.tool_name,
                    "arguments_original": copy.deepcopy(tool_call.arguments),
                    "arguments_processed": {},
                    "can_execute": False,
                    "reason": "",
                    "simulated_output": None,
                }

                if not self.tool_manager.tool_exists(tool_call.tool_name):
                    check_result["reason"] = f"Tool '{tool_call.tool_name}' not found."
                    step_executable = False
                    executability_checks.append(check_result)
                    continue

                tool_schema = self.tool_manager.get_tool_schema(tool_call.tool_name)
                if not tool_schema:  # Should not happen if tool_exists is true
                    check_result["reason"] = f"Schema not found for tool '{tool_call.tool_name}'."
                    step_executable = False
                    executability_checks.append(check_result)
                    continue

                try:
                    processed_args = self._process_placeholders(
                        tool_call.arguments, current_execution_context
                    )
                    check_result["arguments_processed"] = processed_args

                    # Validate required arguments after placeholder processing
                    for req_arg in tool_schema.get("parameters", {}).get("required", []):
                        if req_arg not in processed_args:
                            # If still missing, it's an error (placeholder didn't resolve or wasn't there)
                            raise ValueError(
                                f"Missing required argument '{req_arg}' after placeholder processing."
                            )

                    # Simulate tool invocation using the ToolManager
                    # The new ToolManager's invoke_tool uses an LLM to simulate.
                    tool_output = self.tool_manager.invoke_tool(
                        tool_call.tool_name,
                        processed_args,  # Pass processed arguments
                    )

                    check_result["can_execute"] = True
                    check_result["reason"] = "Successfully simulated invocation."
                    check_result["simulated_output"] = tool_output

                    # Store output for context. Key by tool_name.output for clarity.
                    # Example: {"search_event.output": {"date": "2025-10-15", ...}}
                    # This structure helps in _process_placeholders if it expects such keys.
                    # The current _process_placeholders expects keys like "tool_name.output.field_name"
                    # so we should store the output directly under "tool_name.output"
                    if isinstance(tool_output, dict):  # Assuming tool outputs are dicts
                        step_results_for_context[f"{tool_call.tool_name}.output"] = tool_output
                    else:  # Simple string or other type
                        step_results_for_context[f"{tool_call.tool_name}.output"] = {
                            "value": tool_output
                        }

                except Exception as e:
                    check_result["reason"] = f"Execution/Validation failed: {str(e)}"
                    step_executable = False
                    executability_checks.append(check_result)

            if not step_executable:
                is_overall_executable = False
                break
            else:
                current_execution_context.update(step_results_for_context)

        is_valid_format = not bool(format_errors)
        overall_passed = is_valid_format and is_overall_executable

        return ValidationResult(
            is_valid_format=is_valid_format,
            format_errors=format_errors if format_errors else None,
            is_executable=is_overall_executable,
            executability_checks=executability_checks,
            overall_validation_passed=overall_passed,
        )

    def _get_llm_review_and_feedback(
        self,
        blueprint_candidate: BlueprintCandidate,
        validation_result: ValidationResult,
    ) -> Optional[LLMReview]:
        blueprint_str = blueprint_candidate.blueprint.model_dump_json(indent=2)
        validation_str = validation_result.model_dump_json(indent=2)
        reasoning_str = blueprint_candidate.generation_reasoning or "Not provided."

        system_prompt = """You are an expert in data quality control for AI agent development. Please carefully review the provided task blueprint, the LLM's reasoning during generation, and the automatic validation results.

Your goal is to evaluate whether this blueprint is suitable for generating high-quality training data, and to provide specific, actionable feedback for improvement.

Note: The use of placeholders (`{{tool_name.output.field_name}}`) is essential for multi-step and dependency implementations. If placeholders are used correctly according to the rules, do not penalize for this. In fact, correct use of placeholders may be evaluated positively.

Also, if the following prompt rules are followed, you may evaluate it as high-quality data.
"""

        user_prompt = f"""Below is the task blueprint and related information to be reviewed:

1. **User's Initial Request (q)**:
```
{blueprint_candidate.blueprint.q}
```

2. **Generated Task Blueprint (a_gt_steps, o_gt)**:
```json
{blueprint_str}
```

3. **LLM's Reasoning During Blueprint Generation**:
```
{reasoning_str}
```

4. **Automatic Validation Results (Format and Executability)**:
```json
{validation_str}
```

**Review Items and Evaluation Criteria:**
* **Clarity and Realism of Request (q)**
* **Logical Coherence and Accuracy (a_gt_steps)** (order, parallelism, placeholder usage)
* **Appropriateness of Tool Usage** (selected tools, argument values)
* **Appropriateness of Outcome (o_gt)**
* **Implications of Automatic Validation Results**
* **Overall Quality**: (Select one of Excellent, Good, Fair, Poor)

**Output Format:**
Follow the JSON format of the Pydantic model `LLMReview`. `quality_assessment` is required, and if "Poor" or "Fair", please write `feedback_summary` and `suggested_corrections` specifically.

```json
{{
  "quality_assessment": "...",
  "feedback_summary": "...",
  "suggested_corrections": "..."
}}
```
"""

        try:
            # LLMClient.chat returns (response_text, reasoning_text)
            # We expect response_text to be the JSON string for LLMReview
            response_obj, reasoning_text = self.llm.json_output(
                system_prompt=system_prompt,
                prompt=user_prompt,
                schema=LLMReview,
            )
            # print(f"LLM Review Reasoning: {reasoning_text}")
            print(f"LLM Review Response: {response_obj}")

            # Convert the dictionary to a LLMReview object
            llm_review = LLMReview(**response_obj)
            return llm_review
        except Exception as e:
            print(
                f"Unexpected error during LLM review for q: {blueprint_candidate.blueprint.q}: {e}"
            )
            return None

    def generate_verified_blueprint(
        self, q: str, max_attempts: int = 3
    ) -> Optional[VerifiedBlueprint]:
        if not q.strip():
            print("Warning: Empty query provided. Skipping.")
            return None

        print(f'\n--- Starting Phase 1 for Query: "{q}" ---')

        current_blueprint_candidate: Optional[BlueprintCandidate] = None
        validation_result: Optional[ValidationResult] = None
        llm_review_history: List[LLMReview] = []
        last_feedback: Optional[str] = None
        previous_llm_reviews: List[LLMReview] = []

        for attempt in range(1, max_attempts + 1):
            print(f"\nAttempt {attempt}/{max_attempts}...")

            print("Step 1: Generating blueprint candidate...")
            current_blueprint_candidate = self._generate_blueprint_candidate(
                q,
                previous_feedback=last_feedback,
                previous_llm_reviews=previous_llm_reviews if previous_llm_reviews else None,
                previous_validation_result=validation_result
                if validation_result and not validation_result.overall_validation_passed
                else None,
            )

            if not current_blueprint_candidate:
                print("Failed to generate blueprint candidate.")
                if attempt == max_attempts:
                    break
                last_feedback = "LLM failed to generate a valid blueprint. Please check the prompt and try again."
                continue

            print(f"Blueprint candidate generated for q: {current_blueprint_candidate.blueprint.q}")

            print("Step 2: Validating blueprint format and executability...")
            validation_result = self._validate_blueprint_format_and_executability(
                current_blueprint_candidate.blueprint
            )

            if not validation_result.overall_validation_passed:
                print("Blueprint validation failed.")
                feedback_parts = []
                if not validation_result.is_valid_format and validation_result.format_errors:
                    feedback_parts.append(
                        f"Format errors: {'; '.join(validation_result.format_errors)}"
                    )
                if not validation_result.is_executable:
                    failed_checks_str = "; ".join(
                        [
                            f"Step {chk['step_index']}-Tool {chk['tool_name']}: {chk['reason']}"
                            for chk in validation_result.executability_checks
                            if not chk.get("can_execute")
                        ]
                    )
                    feedback_parts.append(f"Executability issues: {failed_checks_str}")

                last_feedback = (
                    "Automatic validation failed. "
                    + " ".join(feedback_parts)
                    + " Please fix this issue and revise the blueprint."
                )

                if attempt == max_attempts:
                    break
                continue

            print("Blueprint validation successful.")

            print("Step 3: Getting LLM review...")
            llm_review = self._get_llm_review_and_feedback(
                current_blueprint_candidate, validation_result
            )

            if not llm_review:
                # Could be None if LLM fails or returns unparsable review
                print(
                    "Failed to get LLM review or review was unparsable. Assuming 'Fair' quality and requesting refinement."
                )
                last_feedback = "Failed to receive LLM review or the review format is incorrect. Please review and improve the blueprint again."
                # Optionally add a placeholder review to history
                llm_review_history.append(
                    LLMReview(quality_assessment="ReviewFailed", feedback_summary=last_feedback)
                )
                previous_llm_reviews = llm_review_history.copy()
                if attempt == max_attempts:
                    break
                continue

            llm_review_history.append(llm_review)
            previous_llm_reviews = llm_review_history.copy()

            print(f"LLM Review Quality: {llm_review.quality_assessment}")

            if llm_review.quality_assessment in [QualityEnum.Excellent, QualityEnum.Good]:
                print("Blueprint approved by LLM review.")
                return VerifiedBlueprint(
                    blueprint=current_blueprint_candidate.blueprint,
                    validation_result=validation_result,
                    llm_review_history=llm_review_history,
                    generation_attempts=attempt,
                )
            else:
                print("Blueprint requires refinement based on LLM review.")
                last_feedback = (
                    f"LLM review result is '{llm_review.quality_assessment}'. "
                    f"Please improve the blueprint by incorporating the following feedback: "
                    f"{llm_review.feedback_summary or ''} {llm_review.suggested_corrections or ''}"
                )

                if attempt == max_attempts:
                    break
                continue

        print(
            f'--- Failed to generate a verified blueprint for Query: "{q}" after {max_attempts} attempts. ---'
        )
        return None


if __name__ == "__main__":
    import json as json_module
    from datetime import datetime

    print("Initializing APIGen-MT Phase 1 Generator components...")

    # Load configuration from .env file
    from dotenv import load_dotenv

    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE")

    if not api_key:
        print("ERROR: OPENAI_API_KEY not set in .env file. Exiting.")
        exit()

    if not api_base:
        print("ERROR: OPENAI_API_BASE not set in .env file. Exiting.")
        exit()

    # Use LocalOpenAILLMClient for OpenAI-compatible API (NVIDIA)
    llm_client_instance = LocalOpenAILLMClient(
        url=api_base,
        api_key=api_key,
        api_model="nvidia/nemotron-3-super-120b-a12b",
        hf_tokenizer_id=None,  # No tokenizer needed for this client
    )

    # Load tools from BFCL tool pool file
    tool_pool_path = "/home/ishalyminov/data/magnet_mt/output/test_tool_pool.jsonl"
    tool_manager_instance = ToolManager(
        llm=llm_client_instance, tool_pool_path=tool_pool_path
    )

    phase1_generator = APIGenMTPhase1Generator(
        llm_client=llm_client_instance, tool_manager=tool_manager_instance
    )

    print("APIGen-MT Phase 1 Generator initialized with actual components.")

    # Output directory for generated datapoints
    output_dir = "/home/ishalyminov/data/APIGen-MT/data/generated"
    os.makedirs(output_dir, exist_ok=True)

    # Test queries
    queries = [
        "Search for the ADEX 2025 schedule, then register an all-day event named 'ADEX 2025 Preparation Meeting' in the calendar according to the event schedule.",
        # "Tell me Seoul's weather today",  # Simpler query
    ]

    generated_data = []

    for query in queries:
        print(f'\n\n===== Processing Query: "{query}" =====')
        verified_bp = phase1_generator.generate_verified_blueprint(query, max_attempts=3)

        if verified_bp:
            print("\n🎉 Successfully generated Verified Blueprint! 🎉")
            print("\nFinal Blueprint:")
            print(verified_bp.blueprint.model_dump_json(indent=2))
            print("\nValidation Result:")
            print(verified_bp.validation_result.model_dump_json(indent=2))
            print("\nLLM Review History:")
            for review_item in verified_bp.llm_review_history:
                print(review_item.model_dump_json(indent=2))
            print(f"Generated in {verified_bp.generation_attempts} attempt(s).")

            # Store the generated data
            data_point = {
                "query": query,
                "blueprint": verified_bp.blueprint.model_dump(),
                "validation_result": verified_bp.validation_result.model_dump(),
                "llm_review_history": [
                    review.model_dump() for review in verified_bp.llm_review_history
                ],
                "generation_attempts": verified_bp.generation_attempts,
                "timestamp": datetime.now().isoformat(),
            }
            generated_data.append(data_point)
        else:
            print("\n❌ Failed to generate a verified blueprint for the query.")

    # Save generated data to file
    if generated_data:
        output_file = os.path.join(
            output_dir, f"apigen_phase1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        )
        with open(output_file, "w", encoding="utf-8") as f:
            for data_point in generated_data:
                f.write(json_module.dumps(data_point, ensure_ascii=False) + "\n")
        print(f"\n\n✅ Generated {len(generated_data)} datapoint(s) saved to: {output_file}")

    print("\n\n===== Phase 1 Generation Example Finished. =====")