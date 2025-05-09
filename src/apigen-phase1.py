from enum import Enum
from pydantic import BaseModel, Field, ValidationError, field_validator
from typing import List, Dict, Any, Optional
import json
import re
import copy # For deep copying objects
import os
from transformers import AutoTokenizer
from llm_client import LLMClient
from tool_manager import ToolManager

# --- Pydantic Models for APIGen-MT Phase 1 (largely unchanged) ---
class ToolCallInternal(BaseModel):
    tool_name: str = Field(..., description="호출할 도구의 이름")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="도구 호출에 필요한 인자들 (딕셔너리 형태)")

class AGTStep(BaseModel):
    tool_calls: List[ToolCallInternal] = Field(..., description="Ground Truth: 이 단계에서 에이전트가 수행해야 할 도구 호출 리스트")

class Blueprint(BaseModel):
    q: str = Field(..., description="사용자의 초기 질문 또는 요청")
    a_gt_steps: List[AGTStep] = Field(..., description="Ground Truth Steps: 에이전트가 수행해야 할 도구 호출 단계 리스트.")
    o_gt: str = Field(..., description="예상되는 최종 결과에 대한 설명 (문자열)")

    @field_validator('a_gt_steps')
    def check_a_gt_steps_not_empty(cls, v):
        if not v:
            raise ValueError('a_gt_steps는 최소 하나 이상의 단계를 포함해야 합니다.')
        for step in v:
            if not step.tool_calls:
                raise ValueError('a_gt_steps의 각 단계는 최소 하나 이상의 도구 호출을 포함해야 합니다.')
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
    quality_assessment: QualityEnum = Field(..., description="LLM 리뷰의 품질 평가 (Excellent, Good, Fair, Poor 중 하나)")
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

    def _process_placeholders(self, arguments: Dict[str, Any], execution_context: Dict[str, Any]) -> Dict[str, Any]:
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
                            processed_args[arg_name] = arg_value.replace(placeholder_tag, str(current_value))
                    # else:
                        # print(f"Warning: Could not resolve placeholder '{placeholder_full_key}' for argument '{arg_name}'. Kept as is.")
        return processed_args

    def _generate_blueprint_candidate(self, q: str, previous_feedback: Optional[str] = None, previous_llm_reviews: Optional[List[LLMReview]] = None, previous_validation_result: Optional[ValidationResult] = None) -> Optional[BlueprintCandidate]:
        tool_schemas_list = self.tool_manager.get_tools_json_schema()
        tool_schema_json_str = json.dumps(tool_schemas_list, indent=2, ensure_ascii=False)
        
        llm_review_prompt = ""
        if previous_llm_reviews:
            llm_review_prompt += "이전 LLM 리뷰 내역이 있습니다. 아래의 품질 평가와 피드백을 참고하여 청사진을 개선해주십시오.\n"
            for idx, review in enumerate(previous_llm_reviews, 1):
                llm_review_prompt += f"리뷰 {idx}:\n"
                llm_review_prompt += f"- 품질 평가: {review.quality_assessment}\n"
                if review.feedback_summary:
                    llm_review_prompt += f"- 피드백 요약: {review.feedback_summary}\n"
                if review.suggested_corrections:
                    llm_review_prompt += f"- 수정 제안: {review.suggested_corrections}\n"
                llm_review_prompt += "\n"

        validation_prompt = ""
        if previous_validation_result and not previous_validation_result.overall_validation_passed:
            validation_prompt += "이전 자동 검증(validation) 결과가 실패하였습니다. 아래의 검증 결과를 참고하여 청사진을 개선해주십시오.\n"
            if previous_validation_result.format_errors:
                validation_prompt += f"- 형식 오류: {'; '.join(previous_validation_result.format_errors)}\n"
            if not previous_validation_result.is_executable:
                failed_checks = [
                    f"Step {chk['step_index']}-Tool {chk['tool_name']}: {chk['reason']}"
                    for chk in previous_validation_result.executability_checks if not chk.get("can_execute")
                ]
                if failed_checks:
                    validation_prompt += f"- 실행 불가 사유: {'; '.join(failed_checks)}\n"
            validation_prompt += "\n"

        system_prompt = f"""당신은 APIGen-MT 파이프라인의 1단계를 위한 '작업 청사진 생성자'입니다. 현실적이고 검증 가능한 다중 턴 상호작용 시나리오를 나타내는 상세한 청사진을 JSON 객체로 생성해야 합니다.

주요 목표:
1.  **다중 단계 및 의존성**: 사용자의 요청(`q`)을 해결하기 위해 여러 단계의 도구 호출이 필요하고, 한 도구의 출력이 다음 도구의 입력으로 사용되는 시나리오를 우선적으로 생성합니다.
2.  **병렬 실행 가능성**: `a_gt_steps`의 각 단계(`AGTStep`) 내 `tool_calls` 리스트에는 해당 단계에서 병렬로 실행될 수 있는 도구들을 포함시킬 수 있습니다. (예: 서로 독립적인 정보 조회)
3.  **정확한 도구 사용**: 제공된 도구 설명(`Available Tools and their Schemas`)을 정확히 참조하여 `tool_name`과 `arguments`를 구성해야 합니다. 인자 이름과 타입이 일치해야 합니다.
4.  **플레이스홀더**: 이전 단계 도구의 출력을 참조할 때는 `{{tool_name.output.field_name}}` 형식의 플레이스홀더를 사용하십시오. (예: `{{search_event.output.date}}`) `execution_context`에서 해당 경로로 값을 찾을 수 있도록 정확히 명시해야 합니다.
5.  **자연스러운 `o_gt`**: 모든 `a_gt_steps`가 성공적으로 실행되었다고 가정했을 때, 에이전트가 사용자에게 제공할 최종 응답(`o_gt`)은 자연스럽고, 모든 주요 작업 결과를 요약해야 합니다.

{llm_review_prompt}{validation_prompt}
출력 JSON 형식 (Pydantic 모델 `Blueprint` 준수):
```json
{{
  "q": "사용자 요청 문자열",
  "a_gt_steps": [
    {{"tool_calls": [{{"tool_name": "도구명1", "arguments": {{"인자명": "값"}}}}]}},
    {{"tool_calls": [{{"tool_name": "도구명2", "arguments": {{"인자명": "{{도구명1.output.필드명}}"}}}}]}}
  ],
  "o_gt": "최종 결과에 대한 자연스러운 설명"
}}
```
**사고 과정 (생략 가능, 권장):**
최종 JSON을 생성하기 전에, 당신의 사고 과정 (어떤 시나리오를 구상했고, 왜 특정 도구와 순서, 병렬성을 선택했는지 등)을 `<think>...</think>` 태그 안에 작성할 수 있습니다.

Available Tools and their Schemas:
{tool_schema_json_str}
"""
        user_prompt = f"User Query (q): {q}\n\n"
        if previous_feedback:
            user_prompt += f"이전 시도에 대한 피드백입니다. 이 피드백을 반영하여 청사진을 개선해주십시오:\n{previous_feedback}\n\n"
        user_prompt += "위 가이드라인과 사용 가능한 도구를 참고하여, 이 요청에 대한 Blueprint JSON을 생성해주십시오."

        try:
            parsed_json, reasoning = self.llm.json_output(
                prompt=user_prompt,
                system_prompt=system_prompt, # This detailed system prompt will be used by LLMClient
                schema=Blueprint,
                reasoning=True
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

    def _validate_blueprint_format_and_executability(self, blueprint: Blueprint) -> ValidationResult:
        format_errors: List[str] = [] # Should be caught by Pydantic mostly
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
                    "simulated_output": None
                }

                if not self.tool_manager.tool_exists(tool_call.tool_name):
                    check_result["reason"] = f"Tool '{tool_call.tool_name}' not found."
                    step_executable = False
                    executability_checks.append(check_result)
                    continue

                tool_schema = self.tool_manager.get_tool_schema(tool_call.tool_name)
                if not tool_schema: # Should not happen if tool_exists is true
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
                            raise ValueError(f"Missing required argument '{req_arg}' after placeholder processing.")

                    # Simulate tool invocation using the ToolManager
                    # The new ToolManager's invoke_tool uses an LLM to simulate.
                    tool_output = self.tool_manager.invoke_tool(
                        tool_call.tool_name,
                        processed_args # Pass processed arguments
                    )
                    check_result["can_execute"] = True
                    check_result["reason"] = "Successfully simulated invocation."
                    check_result["simulated_output"] = tool_output
                    
                    # Store output for context. Key by tool_name.output for clarity.
                    # Example: {"search_event.output": {"date": "2025-10-15", ...}}
                    # This structure helps in _process_placeholders if it expects such keys.
                    # The current _process_placeholders expects keys like "tool_name.output.field_name"
                    # so we should store the output directly under "tool_name.output"
                    if isinstance(tool_output, dict): # Assuming tool outputs are dicts
                         step_results_for_context[f"{tool_call.tool_name}.output"] = tool_output
                    else: # Simple string or other type
                         step_results_for_context[f"{tool_call.tool_name}.output"] = {"value": tool_output}


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
            overall_validation_passed=overall_passed
        )

    def _get_llm_review_and_feedback(self, blueprint_candidate: BlueprintCandidate, validation_result: ValidationResult) -> Optional[LLMReview]:
        blueprint_str = blueprint_candidate.blueprint.model_dump_json(indent=2)
        validation_str = validation_result.model_dump_json(indent=2)
        reasoning_str = blueprint_candidate.generation_reasoning or "제공되지 않음."

        system_prompt = """당신은 AI 에이전트 개발을 위한 데이터 품질 관리 전문가입니다. 
제공된 작업 청사진(blueprint), 생성 시 LLM의 사고 과정(reasoning), 그리고 자동 검증 결과(validation_result)를 면밀히 검토해주십시오. 
당신의 목표는 이 청사진이 고품질 학습 데이터를 생성하는 데 적합한지 평가하고, 개선을 위한 구체적이고 실행 가능한 피드백을 제공하는 것입니다.

※ 참고: 플레이스홀더(`{{tool_name.output.field_name}}`)의 사용은 다중 단계 및 의존성 구현을 위한 필수적인 요소입니다. 
플레이스홀더가 규칙에 맞게 사용되었다면, 이에 대해 페널티를 부여하지 마십시오. 
오히려 플레이스홀더를 올바르게 사용한 경우는 긍정적으로 평가해도 됩니다.

또한, 아래의 프롬프트 규칙을 준수하였다면 고품질 데이터라고 평가해도 괜찮습니다.
"""
        user_prompt = f"""다음은 검토할 작업 청사진 및 관련 정보입니다:

1.  **사용자 초기 요청 (q)**:
    ```
    {blueprint_candidate.blueprint.q}
    ```

2.  **생성된 작업 청사진 (a_gt_steps, o_gt)**:
    ```json
    {blueprint_str}
    ```

3.  **청사진 생성 시 LLM의 사고 과정**:
    ```
    {reasoning_str}
    ```

4.  **자동 검증 결과 (형식 및 실행 가능성)**:
    ```json
    {validation_str}
    ```

**검토 항목 및 평가 기준:**
* **요청(q)의 명확성 및 현실성**
* **논리적 일관성 및 정확성 (a_gt_steps)** (순서, 병렬성, 플레이스홀더 사용)
* **도구 사용의 적절성** (선택된 도구, 인자 값)
* **결과(o_gt)의 적절성**
* **자동 검증 결과의 시사점**
* **전반적인 품질**: (Excellent, Good, Fair, Poor 중 선택)

**출력 형식:**
Pydantic 모델 `LLMReview`의 JSON 형식을 따라주십시오. `quality_assessment`는 필수이며, "Poor" 또는 "Fair"인 경우 `feedback_summary`와 `suggested_corrections`를 구체적으로 작성해주십시오.
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
                                                    schema=LLMReview
                                                  )

            # print(f"LLM Review Reasoning: {reasoning_text}")
            print(f"LLM Review Response: {response_obj}")
            
            # Convert the dictionary to a LLMReview object
            llm_review = LLMReview(**response_obj)
            return llm_review
        except Exception as e:
            print(f"Unexpected error during LLM review for q: {blueprint_candidate.blueprint.q}: {e}")
            return None

    def generate_verified_blueprint(self, q: str, max_attempts: int = 3) -> Optional[VerifiedBlueprint]:
        if not q.strip():
            print("Warning: Empty query provided. Skipping.")
            return None

        print(f"\n--- Starting Phase 1 for Query: \"{q}\" ---")
        
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
                previous_validation_result=validation_result if validation_result and not validation_result.overall_validation_passed else None
            )
            if not current_blueprint_candidate:
                print("Failed to generate blueprint candidate.")
                if attempt == max_attempts: break
                last_feedback = "LLM이 유효한 청사진을 생성하지 못했습니다. 프롬프트를 확인하고 다시 시도해주세요."
                continue
            print(f"Blueprint candidate generated for q: {current_blueprint_candidate.blueprint.q}")

            print("Step 2: Validating blueprint format and executability...")
            validation_result = self._validate_blueprint_format_and_executability(current_blueprint_candidate.blueprint)
            
            if not validation_result.overall_validation_passed:
                print("Blueprint validation failed.")
                feedback_parts = []
                if not validation_result.is_valid_format and validation_result.format_errors:
                    feedback_parts.append(f"Format errors: {'; '.join(validation_result.format_errors)}")
                if not validation_result.is_executable:
                    failed_checks_str = "; ".join([
                        f"Step {chk['step_index']}-Tool {chk['tool_name']}: {chk['reason']}"
                        for chk in validation_result.executability_checks if not chk.get("can_execute")
                    ])
                    feedback_parts.append(f"Executability issues: {failed_checks_str}")
                last_feedback = "자동 검증에 실패했습니다. " + " ".join(feedback_parts) + " 이 문제를 해결하여 청사진을 수정해주세요."
                if attempt == max_attempts: break
                continue
            print("Blueprint validation successful.")

            print("Step 3: Getting LLM review...")
            llm_review = self._get_llm_review_and_feedback(current_blueprint_candidate, validation_result)
            if not llm_review: # Could be None if LLM fails or returns unparsable review
                print("Failed to get LLM review or review was unparsable. Assuming 'Fair' quality and requesting refinement.")
                last_feedback = "LLM 리뷰를 받는데 실패했거나 리뷰 형식이 올바르지 않습니다. 청사진을 다시 한번 검토하고 개선해주세요."
                # Optionally add a placeholder review to history
                llm_review_history.append(LLMReview(quality_assessment="ReviewFailed", feedback_summary=last_feedback))
                previous_llm_reviews = llm_review_history.copy()
                if attempt == max_attempts: break
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
                    generation_attempts=attempt
                )
            else:
                print("Blueprint requires refinement based on LLM review.")
                last_feedback = f"LLM 리뷰 결과 '{llm_review.quality_assessment}'입니다. 다음 피드백을 반영하여 청사진을 개선해주십시오: {llm_review.feedback_summary or ''} {llm_review.suggested_corrections or ''}"
                if attempt == max_attempts: break
                continue
        
        print(f"--- Failed to generate a verified blueprint for Query: \"{q}\" after {max_attempts} attempts. ---")
        return None

if __name__ == "__main__":
    print("Initializing APIGen-MT Phase 1 Generator components...")
    
    # Ensure FRIENDLI_TOKEN is set in your environment
    if not os.getenv("FRIENDLI_TOKEN"):
        print("ERROR: FRIENDLI_TOKEN environment variable not set. Exiting.")
        exit()

    # Use the actual LLMClient and ToolManager
    # You might want to configure api_model and hf_tokenizer_id based on your Friendli setup
    llm_client_instance = LLMClient(
        # url="YOUR_FRIENDLI_ENDPOINT_IF_DIFFERENT", # Optional
        # api_model="YOUR_FRIENDLI_MODEL_SLUG", # e.g., "meta-llama/Llama-2-7b-chat-hf"
        # hf_tokenizer_id="HF_TOKENIZER_FOR_YOUR_MODEL" # e.g., "meta-llama/Llama-2-7b-chat-hf"
    )
    tool_manager_instance = ToolManager(llm=llm_client_instance)

    phase1_generator = APIGenMTPhase1Generator(
        llm_client=llm_client_instance,
        tool_manager=tool_manager_instance
    )
    print("APIGen-MT Phase 1 Generator initialized with actual components.")

    # Test query
    query = "ADEX 2025 일정을 조사한 후, 행사 일정에 맞춰서 캘린더에 'ADEX 2025 준비 회의'라는 이름의 하루 종일 일정을 등록해줘."
    # query = "오늘 서울 날씨 알려줘" # Simpler query

    print(f"\n\n===== Processing Query: \"{query}\" =====")
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
    else:
        print("\n❌ Failed to generate a verified blueprint for the query.")

    print("\n\n===== Phase 1 Generation Example Finished. =====")
