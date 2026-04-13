#!/usr/bin/env python3
"""
Step-by-step blueprint generator for APIGen-MT.
"""

from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import json
import re
import copy
import os
from llm_client import LLMClient, LocalOpenAILLMClient
from tool_manager import ToolManager


class ToolCallWithOutput(BaseModel):
    """A single tool call with its simulated output."""
    tool_name: str
    arguments: Dict[str, Any] = {}
    output: Any = None


class TrajectoryStep(BaseModel):
    """A single step in the conversation trajectory."""
    step_number: int
    tool_calls: List[ToolCallWithOutput] = []
    reasoning: Optional[str] = None


class ConversationTrajectory(BaseModel):
    """Complete conversation trajectory for a datapoint."""
    query: str
    steps: List[TrajectoryStep] = []
    final_response: str
    tools_used: List[str] = []
    categories_used: List[str] = []


class StepByStepDatapoint(BaseModel):
    """Complete datapoint generated step-by-step."""
    trajectory: ConversationTrajectory
    generation_metadata: Dict[str, Any] = {}
    verification_result: Optional[Dict[str, Any]] = None


class VerificationResult(BaseModel):
    """Complete verification result for a generated datapoint."""
    query: str
    tool_relevance_checks: List[Dict[str, Any]] = []
    order_is_correct: bool
    order_verification_details: str = ""
    output_validations: List[Dict[str, Any]] = []
    placeholder_resolution: Dict[str, Any] = {}
    overall_verification_passed: bool
    verification_summary: str = ""


class StepSelectionResult(BaseModel):
    """Result of LLM selecting the next tool/step."""
    tool_name: str
    arguments: Dict[str, Any] = {}
    reasoning: str


class QueryGenerationResult(BaseModel):
    """Result of generating a user query."""
    query: str
    intent: str
    expected_tools: List[str] = []


class StepByStepGenerator:
    """Generator that creates datapoints step-by-step with immediate tool simulation."""

    def __init__(self, llm_client: LLMClient, tool_manager: ToolManager, num_actions: int = 2, validate_outputs: bool = True):
        self.llm = llm_client
        self.tool_manager = tool_manager
        self.num_actions = num_actions
        self.validate_outputs = validate_outputs

    def _get_tool_schemas_str(self, tools_subset: Optional[List[str]] = None) -> str:
        schemas = self.tool_manager.get_tools_json_schema()
        if tools_subset:
            schemas = [s for s in schemas if s['name'] in tools_subset]
        return json.dumps(schemas, indent=2, ensure_ascii=False)

    def _process_placeholders(self, arguments: Dict[str, Any], execution_context: Dict[str, Any]) -> Dict[str, Any]:
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
                            found = False
                            break
                    if found:
                        placeholder_tag = "{{" + placeholder_full_key + "}}"
                        if arg_value == placeholder_tag:
                            processed_args[arg_name] = current_value
                        else:
                            processed_args[arg_name] = arg_value.replace(placeholder_tag, str(current_value))
        return processed_args

    def generate_user_query(self, focus_category: Optional[str] = None, context_hint: Optional[str] = None) -> QueryGenerationResult:
        categories = list(self.tool_manager.get_categories())
        tools_by_cat = {}
        for cat in categories:
            tools = self.tool_manager.get_tools_by_category(cat)
            if tools:
                tools_by_cat[cat] = [t['name'] for t in tools[:10]]

        prompt = """You are generating a realistic user query for testing a tool-calling system.

Generate a natural, realistic user query that would require using multiple tools to fulfill.

Requirements:
1. The query should be specific and actionable
2. It should mention concrete entities (names, IDs, dates, etc.)
3. It should require at least 2-3 tool calls to complete

Available tool categories:
"""
        for cat, tools in tools_by_cat.items():
            prompt += f"\n{cat}:\n"
            for tool in tools[:5]:
                prompt += f"  - {tool}\n"

        if focus_category and focus_category in tools_by_cat:
            prompt += f"\nFocus category: {focus_category}\n"

        if context_hint:
            prompt += f"\nContext: {context_hint}\n"

        prompt += """
Generate a query that is natural and would require tools to fulfill.

Respond with JSON:
{
    "query": "the generated user query",
    "intent": "brief description of what the user wants",
    "expected_tools": ["tool1", "tool2", ...]
}"""

        try:
            response = self.llm.generate([{"role": "user", "content": prompt}])
            response_text = response.strip()

            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            elif response_text.startswith("{"):
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                response_text = response_text[start:end]

            result = json.loads(response_text)
            return QueryGenerationResult(
                query=result.get("query", ""),
                intent=result.get("intent", ""),
                expected_tools=result.get("expected_tools", [])
            )
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return QueryGenerationResult(query="", intent="", expected_tools=[])
        except Exception as e:
            print(f"Error generating query: {e}")
            return QueryGenerationResult(query="", intent="", expected_tools=[])

    def _generate_next_step(self, query: str, trajectory: List[TrajectoryStep], execution_context: Dict[str, Any], expected_tools: List[str]) -> StepSelectionResult:
        trajectory_str = ""
        for i, step in enumerate(trajectory):
            trajectory_str += f"\nStep {i+1}:"
            for tc in step.tool_calls:
                trajectory_str += f"\n  - {tc.tool_name}({json.dumps(tc.arguments)})"
                if tc.output:
                    trajectory_str += f" -> {json.dumps(tc.output)[:200]}"

        tools_used = set()
        for step in trajectory:
            for tc in step.tool_calls:
                tools_used.add(tc.tool_name)

        tools_remaining = [t for t in expected_tools if t not in tools_used]
        if not tools_remaining:
            return StepSelectionResult(tool_name="__FINAL_RESPONSE__", arguments={}, reasoning="All expected tools have been used.")

        tool_schemas_str = self._get_tool_schemas_str(tools_remaining)

        prompt = f"""You are selecting the next tool to call based on the conversation context.

User Query: {query}

Current Trajectory:{trajectory_str}

Available tools (only use tools from this list):
{tool_schemas_str}

Execution context (outputs from previous steps):
{json.dumps(execution_context, indent=2, default=str)[:1000]}

Select the next tool to call.

Respond with JSON:
{{
    "tool_name": "name of the tool to call",
    "arguments": {{"arg1": "value1", ...}},
    "reasoning": "why this tool is the right next step"
}}"""

        try:
            response = self.llm.generate([{"role": "user", "content": prompt}])
            response_text = response.strip()

            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            elif response_text.startswith("{"):
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                response_text = response_text[start:end]

            result = json.loads(response_text)
            return StepSelectionResult(
                tool_name=result.get("tool_name", ""),
                arguments=result.get("arguments", {}),
                reasoning=result.get("reasoning", "")
            )
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return StepSelectionResult(tool_name="", arguments={}, reasoning="")
        except Exception as e:
            print(f"Error generating step: {e}")
            return StepSelectionResult(tool_name="", arguments={}, reasoning="")

    def _simulate_tool_execution(self, tool_name: str, arguments: Dict[str, Any], execution_context: Dict[str, Any]) -> Any:
        processed_args = self._process_placeholders(arguments, execution_context)
        return self.tool_manager.simulate_tool_call(tool_name=tool_name, arguments=processed_args)

    def generate_datapoint(self, focus_category: Optional[str] = None, context_hint: Optional[str] = None) -> Optional[StepByStepDatapoint]:
        print("\n" + "=" * 60)
        print("STEP-BY-STEP DATAPOINT GENERATION")
        print("=" * 60)

        print(f"\n[Step 1/{self.num_actions + 2}] Generating user query...")
        query_result = self.generate_user_query(focus_category, context_hint)
        if not query_result.query:
            print("Failed to generate query")
            return None

        print(f"Query: {query_result.query}")
        print(f"Intent: {query_result.intent}")
        print(f"Expected tools: {query_result.expected_tools}")

        trajectory = []
        execution_context = {}
        steps_completed = 0

        for step_num in range(1, self.num_actions + 1):
            print(f"\n[Step {step_num + 1}/{self.num_actions + 2}] Selecting tool for step {step_num}...")

            step_result = self._generate_next_step(
                query=query_result.query,
                trajectory=trajectory,
                execution_context=execution_context,
                expected_tools=query_result.expected_tools
            )

            if step_result.tool_name == "__FINAL_RESPONSE__":
                print("All expected tools have been used")
                break

            if not step_result.tool_name:
                print("Failed to generate step")
                break

            print(f"Selected tool: {step_result.tool_name}")
            print(f"Arguments: {json.dumps(step_result.arguments)}")
            print(f"Reasoning: {step_result.reasoning[:100]}...")

            print(f"Simulating {step_result.tool_name}...")
            output = self._simulate_tool_execution(
                tool_name=step_result.tool_name,
                arguments=step_result.arguments,
                execution_context=execution_context
            )

            print(f"Output: {str(output)[:200]}...")

            context_key = f"{step_result.tool_name}_output"
            if output and isinstance(output, dict):
                for k, v in output.items():
                    execution_context[f"{step_result.tool_name}_{k}"] = v
            execution_context[context_key] = output

            tool_call = ToolCallWithOutput(tool_name=step_result.tool_name, arguments=step_result.arguments, output=output)
            trajectory_step = TrajectoryStep(step_number=step_num, tool_calls=[tool_call], reasoning=step_result.reasoning)
            trajectory.append(trajectory_step)
            steps_completed += 1

        print(f"\n[Step {steps_completed + 2}/{self.num_actions + 2}] Generating final response...")
        final_response = self._generate_final_response(query_result.query, trajectory, execution_context)
        print(f"Final response: {final_response[:200]}...")

        tools_used = []
        categories_used = set()
        for step in trajectory:
            for tc in step.tool_calls:
                if tc.tool_name not in tools_used:
                    tools_used.append(tc.tool_name)
                    cat = self.tool_manager.get_tool_category(tc.tool_name)
                    if cat:
                        categories_used.add(cat)

        conv_trajectory = ConversationTrajectory(
            query=query_result.query,
            steps=trajectory,
            final_response=final_response,
            tools_used=tools_used,
            categories_used=list(categories_used)
        )

        metadata = {
            "num_actions": steps_completed,
            "focus_category": focus_category,
            "query_intent": query_result.intent,
            "expected_tools": query_result.expected_tools
        }

        print("\n" + "=" * 60)
        print("RUNNING VERIFICATION")
        print("=" * 60)

        # Run full verification
        verification_result = self.run_full_verification(
            query=query_result.query,
            trajectory=trajectory,
            execution_context=execution_context
        )

        # Convert VerificationResult to dict for storage
        verification_dict = verification_result.model_dump()

        # Create the datapoint with verification results
        datapoint = StepByStepDatapoint(
            trajectory=conv_trajectory,
            generation_metadata=metadata,
            verification_result=verification_dict
        )

        print("\n" + "=" * 60)
        print("DATAPOINT GENERATION COMPLETE")
        print("=" * 60)
        print(f"Tools used: {tools_used}")
        print(f"Categories: {categories_used}")
        print(f"Verification: {'PASSED' if verification_result.overall_verification_passed else 'FAILED'}")

        return datapoint

    def _generate_final_response(self, query: str, trajectory: List[TrajectoryStep], execution_context: Dict[str, Any]) -> str:
        actions_summary = []
        for step in trajectory:
            for tc in step.tool_calls:
                actions_summary.append({
                    "tool": tc.tool_name,
                    "arguments": tc.arguments,
                    "output_summary": str(tc.output)[:100] if tc.output else None
                })

        prompt = f"""Based on the following conversation, generate a natural final response.

User Query: {query}

Actions taken:
{json.dumps(actions_summary, indent=2)}

Generate a concise, natural response that summarizes what was accomplished."""

        try:
            response = self.llm.generate([{"role": "user", "content": prompt}])
            return response.strip()
        except Exception as e:
            print(f"Error generating final response: {e}")
            return "I have completed your request."

    # ==================== VERIFICATION METHODS ====================

    def verify_tool_relevance(self, query: str, tool_name: str, step: TrajectoryStep) -> Dict[str, Any]:
        """Verify if a tool is relevant to the query."""
        tool_schema = self.tool_manager.get_tool_schema(tool_name)
        if not tool_schema:
            return {'tool_name': tool_name, 'is_relevant': False, 'relevance_score': 0.0, 'reasoning': 'Tool not found in tool pool'}

        tool_description = tool_schema.get('description', '')
        keywords = set(tool_description.lower().split())
        query_words = set(query.lower().split())
        overlap = len(keywords & query_words)
        relevance_score = min(1.0, overlap / max(1, len(keywords)))

        name_words = set(tool_name.lower().replace('_', ' ').split())
        name_overlap = len(name_words & query_words)

        is_relevant = relevance_score > 0.1 or name_overlap > 0

        reasoning = f"Tool '{tool_name}': score={relevance_score:.2f}, name_match={name_overlap}"
        reasoning += ". Tool appears relevant." if is_relevant else ". Tool may not be directly relevant."

        return {'tool_name': tool_name, 'is_relevant': is_relevant, 'relevance_score': relevance_score, 'reasoning': reasoning}

    def verify_invocation_order(self, query: str, trajectory: List[TrajectoryStep]) -> Dict[str, Any]:
        """Verify if tools were invoked in a logical order."""
        if not trajectory:
            return {'order_is_correct': True, 'order_verification_details': 'No steps to verify'}

        issues = []
        for i, step in enumerate(trajectory):
            for tc in step.tool_calls:
                tool_name = tc.tool_name.lower()
                if 'create' in tool_name or 'update' in tool_name or 'send' in tool_name:
                    if i == 0 and not any(k in tool_name for k in ['create_new', 'send_notification']):
                        issues.append(f"Step {i+1}: {tc.tool_name} might need prior context")

        order_is_correct = len(issues) == 0
        details = "Order appears logical. " + "; ".join(issues) if issues else "No order issues detected."

        return {'order_is_correct': order_is_correct, 'order_verification_details': details}

    def verify_output_consistency(self, tool_name: str, step_number: int, output: Any, expected_type: str, expected_description: str) -> Dict[str, Any]:
        """Verify if a tool's output matches its declared type and description."""
        if output is None:
            return {'tool_name': tool_name, 'step_number': step_number, 'output_type_matches': False, 'output_description_matches': False, 'issues': ['Output is None']}

        issues = []
        output_type_matches = True
        if expected_type:
            expected_type_lower = expected_type.lower()
            output_type = type(output).__name__.lower()
            type_compatible = False
            if 'dict' in expected_type_lower and isinstance(output, dict):
                type_compatible = True
            elif 'list' in expected_type_lower and isinstance(output, list):
                type_compatible = True
            elif 'string' in expected_type_lower and isinstance(output, str):
                type_compatible = True
            elif 'number' in expected_type_lower and isinstance(output, (int, float)):
                type_compatible = True
            elif 'bool' in expected_type_lower and isinstance(output, bool):
                type_compatible = True
            elif expected_type_lower in output_type:
                type_compatible = True
            if not type_compatible:
                output_type_matches = False
                issues.append(f"Type mismatch: expected {expected_type}, got {output_type}")

        output_description_matches = True
        if expected_description and output:
            output_str = str(output).lower()
            desc_words = set(expected_description.lower().split())
            output_words = set(output_str.split())
            overlap = len(desc_words & output_words)
            if overlap < 2 and len(desc_words) > 5:
                output_description_matches = False
                issues.append("Output may not match description")

        return {'tool_name': tool_name, 'step_number': step_number, 'output_type_matches': output_type_matches, 'output_description_matches': output_description_matches, 'issues': issues}

    def verify_placeholder_resolution(self, trajectory: List[TrajectoryStep], execution_context: Dict[str, Any]) -> Dict[str, Any]:
        """Verify that all placeholders in tool arguments were resolved correctly."""
        total_placeholders = 0
        resolved_count = 0
        details = []
        placeholder_pattern = re.compile(r"\{\{([^{}]+)\}\}")

        for step in trajectory:
            for tc in step.tool_calls:
                for arg_name, arg_value in tc.arguments.items():
                    if isinstance(arg_value, str):
                        placeholders = placeholder_pattern.findall(arg_value)
                        for placeholder in placeholders:
                            total_placeholders += 1
                            keys = placeholder.split('.')
                            current = execution_context
                            found = True
                            for key in keys:
                                if isinstance(current, dict) and key in current:
                                    current = current[key]
                                else:
                                    found = False
                                    break
                            if found:
                                resolved_count += 1
                                details.append({'step': step.step_number, 'tool': tc.tool_name, 'argument': arg_name, 'placeholder': f"{{{{{placeholder}}}}}", 'resolved': True, 'resolved_value': str(current)[:100]})
                            else:
                                details.append({'step': step.step_number, 'tool': tc.tool_name, 'argument': arg_name, 'placeholder': f"{{{{{placeholder}}}}}", 'resolved': False, 'resolved_value': None})

        return {'all_resolved': total_placeholders == resolved_count, 'total_placeholders': total_placeholders, 'resolved_count': resolved_count, 'details': details}

    def run_full_verification(self, query: str, trajectory: List[TrajectoryStep], execution_context: Dict[str, Any]) -> VerificationResult:
        """
        Run all verification checks on a generated datapoint.

        Verification checks:
        1. Tool relevance - are tools relevant for the query?
        2. Invocation order - are tools invoked in the right order?
        3. Output consistency - do outputs match tool declarations?
        4. Placeholder resolution - are all placeholders resolved?
        """
        print("\n=== Running Verification ===")

        # 1. Check tool relevance
        tool_relevance_checks = []
        all_relevant = True
        for step in trajectory:
            for tc in step.tool_calls:
                check = self.verify_tool_relevance(query, tc.tool_name, step)
                tool_relevance_checks.append(check)
                if not check['is_relevant']:
                    all_relevant = False
                print(f"  {tc.tool_name}: relevance={check['relevance_score']:.2f}, relevant={check['is_relevant']}")

        # 2. Verify invocation order
        print("Verifying invocation order...")
        order_result = self.verify_invocation_order(query, trajectory)
        print(f"  Order correct: {order_result['order_is_correct']}")
        print(f"  Details: {order_result['order_verification_details']}")

        # 3. Verify output consistency
        print("Verifying output consistency...")
        output_validations = []
        all_outputs_valid = True
        for step in trajectory:
            for tc in step.tool_calls:
                tool_schema = self.tool_manager.get_tool_schema(tc.tool_name)
                expected_type = tool_schema.get('output_type', 'unknown') if tool_schema else 'unknown'
                expected_desc = tool_schema.get('output_description', '') if tool_schema else ''

                validation = self.verify_output_consistency(tc.tool_name, step.step_number, tc.output, expected_type, expected_desc)
                output_validations.append(validation)
                if not validation['output_type_matches'] or not validation['output_description_matches']:
                    all_outputs_valid = False
                print(f"  {tc.tool_name}: type_match={validation['output_type_matches']}, desc_match={validation['output_description_matches']}")
                if validation['issues']:
                    for issue in validation['issues']:
                        print(f"    - {issue}")

        # 4. Check placeholder resolution
        print("Checking placeholder resolution...")
        placeholder_result = self.verify_placeholder_resolution(trajectory, execution_context)
        print(f"  Resolved: {placeholder_result['resolved_count']}/{placeholder_result['total_placeholders']}")

        # Overall check
        overall_passed = all_relevant and order_result['order_is_correct'] and all_outputs_valid and placeholder_result['all_resolved']

        issues = []
        if not all_relevant:
            issues.append("Some tools are not relevant to the query")
        if not order_result['order_is_correct']:
            issues.append("Tool invocation order may be incorrect")
        if not all_outputs_valid:
            issues.append("Some tool outputs don't match their declarations")
        if not placeholder_result['all_resolved']:
            issues.append(f"{placeholder_result['total_placeholders'] - placeholder_result['resolved_count']} placeholders were not resolved")

        summary = "Verification PASSED - all checks successful" if overall_passed else "Verification FAILED - issues found"
        print(f"\n=== Verification Result: {'PASSED' if overall_passed else 'FAILED'} ===")
        if issues:
            for issue in issues:
                print(f"  - {issue}")

        return VerificationResult(
            query=query,
            tool_relevance_checks=tool_relevance_checks,
            order_is_correct=order_result['order_is_correct'],
            order_verification_details=order_result['order_verification_details'],
            output_validations=output_validations,
            placeholder_resolution=placeholder_result,
            overall_verification_passed=overall_passed,
            verification_summary=summary
        )


# --- CLI Entry Point ---

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE")

    if not api_key or not api_base:
        print("ERROR: OPENAI_API_KEY or OPENAI_API_BASE not set")
        exit(1)

    llm_client = LocalOpenAILLMClient(
        url=api_base,
        api_key=api_key,
        api_model="nvidia/nemotron-3-super-120b-a12b",
        hf_tokenizer_id=None
    )

    tool_pool_path = "/home/ishalyminov/data/APIGen-MT/magnet_tool_extraction/bfcl_v3_tools_with_outputs.jsonl"
    tool_manager = ToolManager(llm=llm_client, tool_pool_path=tool_pool_path)

    generator = StepByStepGenerator(
        llm_client=llm_client,
        tool_manager=tool_manager,
        num_actions=2
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