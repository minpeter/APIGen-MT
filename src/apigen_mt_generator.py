from llm_client import LLMClient
from tool_manager import ToolManager
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
    messages: Optional[List[Dict[str, Any]]] = None


class APIGenMTGenerator:
    """
    The APIGenMTGenerator class generates training data for multi-turn API calling tasks.
    
    This class processes externally provided queries (q) to generate:
    - a_gt: Ground truth tool calls sequence
    - o_gt: Expected outcome after executing the tool calls
    - verification_result: Results of actually executing the tool calls
    - llm_review: Optional LLM-generated review of the data quality
    
    It doesn't generate queries internally but relies on queries provided from outside.
    """
    
    def __init__(self, llm: LLMClient, tool_manager: ToolManager):
        """
        Initialize the APIGenMTGenerator.
        
        Args:
            llm: LLMClient for LLM interactions
            tool_manager: ToolManager for handling tools
        """
        self.llm = llm
        self.tool_manager = tool_manager

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


    def _execute_and_verify_a_gt(self, q: str, a_gt: List[ToolCallInternal], o_gt: str) -> Dict[str, Any]:
        executed_trajectory: List[Dict[str, Any]] = []
        conversation_messages: List[Dict[str, Any]] = []
        overall_status = "success" # Assume success until a failure

        # Add initial system message to conversation flow
        conversation_messages.append({
            "role": "system", 
            "content": "You are a helpful assistant that uses tools to fulfill user requests."
        })
        
        # Add initial user message based on the actual query
        conversation_messages.append({
            "role": "user",
            "content": q
        })

        for i, tool_call in enumerate(a_gt):
            try:
                # Using the correct method name: invoke_tool instead of execute_tool
                result = self.tool_manager.invoke_tool(tool_call.tool_name, tool_call.tool_input)
                executed_trajectory.append({
                    "step": i + 1,
                    "tool_name": tool_call.tool_name,
                    "tool_input": tool_call.tool_input,
                    "status": "success",
                    "result": result 
                })
                
                # Add assistant message with tool call to conversation flow
                assistant_message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call_{i}",
                            "type": "function",
                            "function": {
                                "name": tool_call.tool_name,
                                "arguments": json.dumps(tool_call.tool_input)
                            }
                        }
                    ]
                }
                conversation_messages.append(assistant_message)
                
                # Add tool response message to conversation flow
                conversation_messages.append({
                    "role": "tool",
                    "tool_call_id": f"call_{i}",
                    "name": tool_call.tool_name,
                    "content": result
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
        
        # Add final assistant response using the expected outcome
        if overall_status == "success":
            conversation_messages.append({
                "role": "assistant",
                "content": o_gt
            })
        
        # Only return verification result without messages in the verification_result
        result = {
            "overall_status": overall_status, 
            "trajectory": executed_trajectory
        }
            
        return result

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
        
    def generate_for_query(self, q: str, with_review: bool = True) -> Optional[GeneratedData]:
        """
        Generate data for a single query.
        
        Args:
            q: The query to generate data for
            with_review: Whether to include LLM review of the generated data
            
        Returns:
            GeneratedData object or None if generation failed
        """
        if not q.strip():
            print("Warning: Empty or whitespace-only query. Skipping.")
            return None
            
        print(f"\n--- Processing query: {q} ---")
        
        # 1. Generate A_GT and O_GT based on Q
        generation_result = self._generate_a_gt_o_gt(q)
        if not generation_result:
            print(f"Failed to generate a_gt and o_gt for query: {q}.")
            return None
            
        a_gt, o_gt = generation_result
        
        # 2. Execute and Verify A_GT - also store conversation messages
        conversation_messages: List[Dict[str, Any]] = []
        
        # Add initial system message to conversation flow
        conversation_messages.append({
            "role": "system", 
            "content": "You are a helpful assistant that uses tools to fulfill user requests."
        })
        
        # Add initial user message based on the actual query
        conversation_messages.append({
            "role": "user",
            "content": q
        })
        
        # Execute tools and get verification result
        verification_result = self._execute_and_verify_a_gt(q, a_gt, o_gt)
        
        # Populate conversation messages for successful verifications
        if verification_result.get("overall_status") == "success":
            # Add messages for each tool call
            for i, tool_call in enumerate(a_gt):
                # Add assistant message with tool call
                assistant_message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call_{i}",
                            "type": "function",
                            "function": {
                                "name": tool_call.tool_name,
                                "arguments": json.dumps(tool_call.tool_input)
                            }
                        }
                    ]
                }
                conversation_messages.append(assistant_message)
                
                # Get the result from the verification trajectory
                result = None
                for step in verification_result.get("trajectory", []):
                    if step.get("step") == i + 1 and step.get("status") == "success":
                        result = step.get("result")
                        break
                
                # Add tool response message
                conversation_messages.append({
                    "role": "tool",
                    "tool_call_id": f"call_{i}",
                    "name": tool_call.tool_name,
                    "content": result
                })
            
            # Add final assistant response
            conversation_messages.append({
                "role": "assistant",
                "content": o_gt
            })
        
        # 3. (Optional) Review with LLM
        llm_review = None
        if with_review:
            try:
                llm_review = self._review_blueprint_with_llm(q, a_gt, o_gt, verification_result)
            except Exception as e:
                print(f"Error during LLM review for query: {q}: {e}")
                # Continue without review if it fails

        # Prepare data for GeneratedData object
        data_kwargs = {
            "q": q,
            "a_gt": a_gt,
            "o_gt": o_gt,
            "verification_result": verification_result,
            "llm_review": llm_review
        }
        
        # Add messages to the data if verification was successful
        if verification_result.get("overall_status") == "success":
            data_kwargs["messages"] = conversation_messages
            print("Added conversation messages to generated data.")
        
        generated_data = GeneratedData(**data_kwargs)
        print(f"--- Finished processing query ---")
        
        return generated_data
        
    def generate(self, queries: List[str], with_review: bool = True) -> List[GeneratedData]:
        """
        Generate data for multiple queries.
        
        Args:
            queries: List of queries to process
            with_review: Whether to include LLM review of the generated data
            
        Returns:
            List of GeneratedData objects
        """
        all_generated_data: List[GeneratedData] = []
        
        print(f"Processing {len(queries)} queries...")
        
        for i, q in enumerate(queries):
            print(f"\n--- Processing Q {i+1}/{len(queries)}: {q} ---")
            generated_data = self.generate_for_query(q, with_review)
            
            if generated_data:
                all_generated_data.append(generated_data)
                print(f"--- Finished processing Q {i+1}/{len(queries)} ---")
            else:
                print(f"Failed to generate data for query: {q}")

        if not all_generated_data:
            print("No data was successfully generated for any of the queries.")
        
        return all_generated_data

if __name__ == "__main__":
    # --- Instantiate actual components ---
    # Ensure you have your API key set as an environment variable or directly here
    # For example: api_key = os.getenv("OPENAI_API_KEY") or "your_actual_api_key"
    # Ensure your tool configuration path is correct for ToolManager
    # For example: tool_config_path = "path/to/your/tools_config.json"

    print("Starting APIGenMTGenerator with actual components...")

    try:
        # 1. Initialize LLMClient
        llm_client = LLMClient() 
        print("LLMClient initialized.")

        # 2. Initialize ToolManager
        tool_manager = ToolManager(
            llm=llm_client,
        )
        print("ToolManager initialized.")

        # 3. Initialize APIGenMTGenerator with components
        generator = APIGenMTGenerator(
            llm=llm_client,
            tool_manager=tool_manager
        )
        print("APIGenMTGenerator initialized with actual components.")

        # Example 1: Generate data for a single query
        print("\n=== Example 1: Generate data for a single query ===")
        # single_query = "What's my schedule look like on June 7th after 1 PM?"
        
        single_query = "Search for volunteer opportunities in Austin and block time for Habitat for Humanity on September 21st."
        single_result = generator.generate_for_query(single_query)
        
        if single_result:
            print("\n--- Single Query Result ---")
            print(single_result.model_dump_json(indent=2))
        else:
            print("\nFailed to generate data for the single query.")

        # # Example 2: Generate data for multiple predefined queries
        # print("\n=== Example 2: Generate data for multiple predefined queries ===")
        # predefined_queries = [
        #     "Send an email to John about the meeting tomorrow",
        #     "Set a reminder to buy groceries at 6 PM",
        #     "What time is my first meeting tomorrow?"
        # ]
        # predefined_results = generator.generate(queries=predefined_queries)
        
        # if predefined_results:
        #     print(f"\n--- Predefined Queries Results - {len(predefined_results)} items generated ---")
        #     for i, instance in enumerate(predefined_results):
        #         print(f"\n--- Item {i+1}/{len(predefined_results)} ---")
        #         print(instance.model_dump_json(indent=2))
        # else:
        #     print("\nFailed to generate data for predefined queries.")

    except ImportError as e:
        print(f"Error: Missing one of the core modules (LLMClient, ToolManager, QGenerator): {e}")
        print("Please ensure these modules are correctly defined and importable in your PYTHONPATH.")
    except Exception as e:
        print(f"An error occurred during execution with actual components: {e}")
        import traceback
        traceback.print_exc()

    print("\nExample finished.")
