from llm_client import LLMClient
from tool_manager import ToolManager
from q_generator import QGenerator
from pydantic import BaseModel, ValidationError
from typing import List, Dict, Any, Optional, Tuple
import json
import re 


class ToolCallInternal(BaseModel):
    tool_name: str
    tool_input: Dict[str, Any]


class GeneratedData(BaseModel):
    q: str
    a_gt: List[ToolCallInternal]
    o_gt: str
    verification_result: Dict[str, Any]
    llm_review: Optional[str] = None


class APIGenMTGenerator:
    def __init__(self, llm: LLMClient, tool_manager: ToolManager, q_generator: QGenerator):
        self.llm = llm
        self.tool_manager = tool_manager
        self.q_generator = q_generator

    def _extract_json_from_markdown(self, md_text: str) -> Optional[Dict[str, Any]]:
        match = re.search(r"```json\s*([\s\S]*?)\s*```", md_text, re.IGNORECASE)
        if match:
            json_str = match.group(1)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                print(f"Problematic JSON string: {json_str}")
                return None
        # Fallback: try to parse the whole string if no markdown block is found
        try:
            return json.loads(md_text)
        except json.JSONDecodeError:
            pass # It's okay if this fails, it means it wasn't plain JSON
        
        print("No JSON block found in markdown or plain JSON.")
        return None

    def _call_llm_for_blueprint_parts(self, q: str, tool_schema_json: str) -> Optional[Dict[str, Any]]:
        system_prompt = f'''You are an expert AI assistant. Your task is to generate a sequence of tool calls (a_gt) and an expected outcome (o_gt) for a given user query (q).
You MUST respond in JSON format, enclosed in a markdown code block (```json ... ```).
The JSON object should have two top-level keys: "tool_calls" and "expected_outcome".
- "tool_calls": A list of objects, where each object has "tool_name" (string) and "tool_input" (a JSON object of parameters).
- "expected_outcome": A string describing the final result or state after executing the tool calls.

Available Tools and their Schemas:
{tool_schema_json}
'''
        user_prompt = f'''User Query (q): {q}

Generate the tool calls and expected outcome based on this query and the available tools.
Ensure your entire response is a single JSON object within a markdown code block.
'''
        
        # Use the chat method of LLMClient with proper message structure
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Using a low temperature for more deterministic output for generation tasks
        raw_llm_response, _ = self.llm.chat(messages=messages, kwargs={"temperature": 0.1})
        
        if raw_llm_response:
            return self._extract_json_from_markdown(raw_llm_response)
        return None

    def _generate_a_gt_o_gt(self, q: str) -> Optional[Tuple[List[ToolCallInternal], str]]:
        tool_schema_json = self.tool_manager.get_tools_json_schema() # Assuming this method exists
        if not tool_schema_json:
            print("Error: Could not retrieve tool schema.")
            return None

        blueprint_dict = self._call_llm_for_blueprint_parts(q, tool_schema_json)

        if not blueprint_dict:
            print(f"Failed to get blueprint from LLM for q: {q}")
            return None

        try:
            # Validate tool_calls part
            raw_tool_calls = blueprint_dict.get("tool_calls")
            if not isinstance(raw_tool_calls, list):
                print(f"LLM response 'tool_calls' is not a list for q: {q}")
                return None
            
            parsed_a_gt = [ToolCallInternal(**call) for call in raw_tool_calls]
            
            # Validate expected_outcome part
            o_gt = blueprint_dict.get("expected_outcome")
            if not isinstance(o_gt, str):
                print(f"LLM response 'expected_outcome' is not a string for q: {q}")
                return None
                
            return parsed_a_gt, o_gt
        except ValidationError as e:
            print(f"Pydantic validation error for blueprint from LLM for q: {q}: {e}")
            print(f"Raw blueprint_dict: {blueprint_dict}")
            return None
        except Exception as e:
            print(f"Unexpected error during parsing/validation for q: {q}: {e}")
            print(f"Raw blueprint_dict: {blueprint_dict}")
            return None


    def _execute_and_verify_a_gt(self, a_gt: List[ToolCallInternal]) -> Dict[str, Any]:
        executed_trajectory: List[Dict[str, Any]] = []
        overall_status = "success" # Assume success until a failure

        for i, tool_call in enumerate(a_gt):
            try:
                # Assuming tool_manager.execute_tool returns the result of the tool execution
                # This might be a string, dict, or any other type depending on the tool
                result = self.tool_manager.execute_tool(tool_call.tool_name, tool_call.tool_input)
                executed_trajectory.append({
                    "step": i + 1,
                    "tool_name": tool_call.tool_name,
                    "tool_input": tool_call.tool_input,
                    "status": "success",
                    "result": result 
                })
            except Exception as e:
                overall_status = "error"
                executed_trajectory.append({
                    "step": i + 1,
                    "tool_name": tool_call.tool_name,
                    "tool_input": tool_call.tool_input,
                    "status": "error",
                    "error_message": str(e)
                })
                # Decide whether to break on error or continue executing subsequent tools
                # For now, let's break as subsequent tools might depend on the failed one.
                break 
        
        return {"overall_status": overall_status, "trajectory": executed_trajectory}

    def _review_blueprint_with_llm(self, q: str, a_gt: List[ToolCallInternal], o_gt: str, verification_result: Dict[str, Any]) -> Optional[str]:
        a_gt_json_str = json.dumps([call.model_dump() for call in a_gt], indent=2)
        verification_result_str = json.dumps(verification_result, indent=2)

        system_prompt = "You are an expert AI assistant. Your task is to review a generated plan (q, a_gt, o_gt) and its execution trace."
        user_prompt = f'''
Original Query (q):
{q}

Generated Tool Calls (a_gt):
{a_gt_json_str}

Generated Expected Outcome (o_gt):
{o_gt}

Actual Execution Trace and Verification Result:
{verification_result_str}

Please review the following aspects:
1.  Clarity and relevance of 'q'.
2.  Logical coherence and correctness of 'a_gt' to address 'q'.
3.  Appropriateness and feasibility of tool calls in 'a_gt' given typical tool functionalities.
4.  Realism and achievability of 'o_gt'.
5.  Assessment of the execution trace: Did tools execute as expected? Were there errors?
6.  Alignment: Does the final state suggested by the execution trace align with 'o_gt'?
7.  Overall quality: Provide an overall assessment (e.g., Excellent, Good, Fair, Poor) and concise suggestions for improvement if any.

Provide your review as a structured text.
'''
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        review, _ = self.llm.chat(messages=messages, kwargs={"temperature": 0.5}) # Higher temp for more nuanced review
        return review

    def generate(self) -> List[GeneratedData]:
        all_generated_data: List[GeneratedData] = []

        # 1. Generate a batch of Qs
        try:
            # QGenerator.generate_batch_qs takes (num_to_generate, existing_qs_list)
            # If topic is provided, we'll add it as a starter question to guide generation
            existing_qs = []

            batch_qs_objects = self.q_generator.generate_batch_qs(
                num_to_generate=20,
                existing_qs_list=existing_qs,
                llm_batch_size=15,
                # llm_overgen_factor=1.8, # This argument is no longer used
                similarity_threshold=0.85,
                top_n_similar=3
            )
        except AttributeError:
            print("Error: self.q_generator.generate_batch_qs method not found in QGenerator.")
            print("Please ensure QGenerator has the method 'generate_batch_qs(num_to_generate, existing_qs_list, ...)'.")
            return []
        except Exception as e:
            print(f"Error calling self.q_generator.generate_batch_qs: {e}")
            return []

        if not batch_qs_objects:
            print(f"No questions generated by q_generator.generate_batch_qs.")
            return []

        print(f"Generated {len(batch_qs_objects)} questions. Processing each...")

        for i, q_obj in enumerate(batch_qs_objects):
            current_q = ""
            if isinstance(q_obj, dict) and "q" in q_obj:
                current_q = q_obj["q"]
            else:
                print(f"Warning: Unexpected item type from generate_batch_qs at index {i}: {type(q_obj)}. Expected dict with 'q' key. Skipping.")
                continue
            
            if not current_q.strip(): # Skip if the question is empty or whitespace only
                print(f"Warning: Empty or whitespace-only question string at index {i}. Skipping.")
                continue

            print(f"\n--- Processing Q {i+1}/{len(batch_qs_objects)}: {current_q} ---")

            # 2. Generate A_GT and O_GT based on Q
            generation_result = self._generate_a_gt_o_gt(current_q)
            if not generation_result:
                print(f"Failed to generate a_gt and o_gt for q: {current_q}. Skipping this Q.")
                continue
            a_gt, o_gt = generation_result
            
            # 3. Execute and Verify A_GT
            verification_result = self._execute_and_verify_a_gt(a_gt)
            
            # 4. (Optional) Review with LLM
            llm_review = None
            try:
                llm_review = self._review_blueprint_with_llm(current_q, a_gt, o_gt, verification_result)
            except Exception as e:
                print(f"Error during LLM review for q: {current_q}: {e}")
                # Continue without review if it fails

            generated_instance = GeneratedData(
                q=current_q,
                a_gt=a_gt,
                o_gt=o_gt,
                verification_result=verification_result,
                llm_review=llm_review
            )
            all_generated_data.append(generated_instance)
            print(f"--- Finished processing Q {i+1}/{len(batch_qs_objects)} ---")

        if not all_generated_data:
            print("No data was successfully generated for any of the questions.")
        
        return all_generated_data

if __name__ == "__main__":
    # --- Instantiate actual components ---
    # Ensure you have your API key set as an environment variable or directly here
    # For example: api_key = os.getenv("OPENAI_API_KEY") or "your_actual_api_key"
    # Ensure your tool configuration path is correct for ToolManager
    # For example: tool_config_path = "path/to/your/tools_config.json"

    print("Starting APIGenMTGenerator with actual components...")

    try:
        llm_client = LLMClient() 
        print("LLMClient initialized.")

        tool_manager = ToolManager(
            llm=llm_client,
        )
        print("ToolManager initialized.")

        # 3. Initialize QGenerator
        q_generator = QGenerator(tool_manager=tool_manager, llm=llm_client)
        print("QGenerator initialized.")

        # 4. Initialize APIGenMTGenerator with actual components
        generator = APIGenMTGenerator(
            llm=llm_client,
            tool_manager=tool_manager,
            q_generator=q_generator
        )
        print("APIGenMTGenerator initialized with actual components.")
 
        generated_data_instance = generator.generate()

        if generated_data_instance:
            print("\n--- Generated Data (Actual Components) ---")
            print(generated_data_instance.model_dump_json(indent=2))
        else:
            print("\nFailed to generate data with actual components.")

    except ImportError as e:
        print(f"Error: Missing one of the core modules (LLMClient, ToolManager, QGenerator): {e}")
        print("Please ensure these modules are correctly defined and importable in your PYTHONPATH.")
    except Exception as e:
        print(f"An error occurred during execution with actual components: {e}")
        import traceback
        traceback.print_exc()

    print("\nExample finished.")
