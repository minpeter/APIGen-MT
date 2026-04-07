from llm_client import LLMClient
from tool_manager import ToolManager
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, Any, Optional, Tuple, Union
import json
import re

class ToolCallInternal(BaseModel):
    tool_name: str = Field(..., description="Name of the tool to call")
    arguments: Dict[str, Any] = Field(..., description="Arguments required for the tool call (dictionary format)")

class aGTStep(BaseModel):
    a_gt: List[ToolCallInternal] = Field(..., description="Ground Truth: Tool calls that the agent should perform")

class Blueprint(BaseModel):
    q: str = Field(..., description="User's initial question or request")
    a_gt_steps: List[aGTStep] = Field(..., description="Ground Truth Steps: A list representing the order of tool calls the agent should perform. Tools in the same step are executed in parallel, and should consist of logically executable tools.")
    o_gt: str = Field(..., description="Description of the expected final result (string)")

class GeneratedData(BaseModel):
    q: str
    a_gt_steps: List[aGTStep]
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
    - messages: Optional list of messages exchanged during the process
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

    def _generate_blueprint(self, q: str) -> Optional[Tuple[List[aGTStep], str]]:
        """
        Generate a blueprint for the given query.
        a_gt_steps, and o_gt are generated based on the query.
        """
        tool_schema_json = self.tool_manager.get_tools_json_schema() # Assuming this method exists
        if not tool_schema_json:
            print("Error: Could not retrieve tool schema.")
            return None

        system_prompt = f'''You are the 'Task Blueprint Generator' for Step 1 of the APIGen-MT pipeline. Your goal is to create a detailed blueprint representing a **realistic and verifiable multi-turn interaction scenario** between a user and an AI agent. Based on the provided tool descriptions, you need to generate a **JSON object** with the following components:

1. `q` (string): The user's initial question/request. **Should be specific and natural**, and it's good to represent a scenario that **may require multiple steps of interaction**, such as the agent retrieving information, changing state, and confirming back to the user.
2. `a_gt_steps` (list): The **Ground Truth tool call list** that the agent should call to completely and in correct order resolve the user's request `q`. Each element should be in the format `{{"tool_name": "tool_name", "arguments": {{"arg_name": "value", ...}}}}`. Must match the available tool descriptions. Must include **at least 1 or more tool calls**, and if possible, create a scenario that requires **2 or more sequential tool calls**.
   - If the result of a previous tool call needs to be used in the next tool call, use a placeholder in the format `{{tool_name.output.field_name}}` for the argument value.
3. `o_gt` (string): A **natural description of the result summary or response message** that the agent should finally provide to the user, assuming all `a_gt_steps` were successfully executed. Should reflect all tool call results.

**Thinking Process (Optional but Recommended):**
- Before generating the final JSON, you can **write your thinking process (what scenario you envisioned, why you chose certain tools and order, etc.) inside ` HDC...DCH` tags**.

Available Tools and their Schemas:
{tool_schema_json}
'''
        user_prompt = f'''User Query (q): {q}

Generate the blueprint for this query, including steps for multi-turn interaction if needed.
Remember to properly structure a_gt_steps for parallel or sequential tool calls.
'''
        
        try:
            # Use json_output method of LLMClient with Blueprint schema for structured output
            parsed_blueprint, _ = self.llm.json_output(
                prompt=user_prompt,
                system_prompt=system_prompt,
                schema=Blueprint,
                reasoning=True
            )
        except Exception as e:
            print(f"Error generating blueprint: {e}")
            print(f"Failed to get blueprint from LLM for q: {q}")
            return None
        
        try:
            # Extract the 'a_gt_steps' and 'o_gt' from the parsed blueprint
            a_gt_steps = parsed_blueprint.get("a_gt_steps")
            if not isinstance(a_gt_steps, list):
                print(f"LLM response 'a_gt_steps' is not a list for q: {q}")
                return None
            
            # Validate each step in a_gt_steps
            parsed_steps = []
            for step_data in a_gt_steps:
                # Check if the step has 'a_gt' key and it's a list
                a_gt = step_data.get("a_gt")
                if not isinstance(a_gt, list):
                    print(f"Step 'a_gt' is not a list in a_gt_steps for q: {q}")
                    return None
                
                # Validate each tool call within the step
                tool_calls = []
                for tool_call in a_gt:
                    # Convert 'tool_input' to 'arguments' if needed for backward compatibility
                    if "tool_name" in tool_call and "tool_input" in tool_call and "arguments" not in tool_call:
                        tool_call["arguments"] = tool_call.pop("tool_input")
                    
                    # Process placeholders in arguments
                    arguments = tool_call.get("arguments", {})
                    for arg_name, arg_value in arguments.items():
                        if isinstance(arg_value, str) and "{{" in arg_value and "}}" in arg_value:
                            # This is a placeholder, we keep it as is for now
                            # It will be replaced during execution time
                            pass
                    
                    tool_calls.append(ToolCallInternal(
                        tool_name=tool_call.get("tool_name"),
                        arguments=arguments
                    ))
                
                parsed_steps.append(aGTStep(a_gt=tool_calls))
                
            # Validate expected_outcome part
            o_gt = parsed_blueprint.get("o_gt")
            if not isinstance(o_gt, str):
                print(f"LLM response 'o_gt' is not a string for q: {q}")
                return None
                
            return parsed_steps, o_gt
        except ValidationError as e:
            print(f"Pydantic validation error for blueprint from LLM for q: {q}: {e}")
            print(f"Raw blueprint: {parsed_blueprint}")
            return None
        except Exception as e:
            print(f"Unexpected error during parsing/validation for q: {q}: {e}")
            print(f"Raw blueprint: {parsed_blueprint}")
            return None
    
    def _execute_and_verify_a_gt_steps(self, q: str, a_gt_steps: List[aGTStep], o_gt: str) -> Dict[str, Any]:
        """
        Execute and verify the ground truth steps.
        
        Args:
            q: Original query
            a_gt_steps: List of steps with tool calls to execute
            o_gt: Expected outcome
            
        Returns:
            Verification result dictionary
        """
        executed_trajectory: List[Dict[str, Any]] = []
        conversation_messages: List[Dict[str, Any]] = []
        overall_status = "success"  # Assume success until a failure
        execution_context = {}  # Store results from previous tool calls for placeholder substitution
        
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
        
        # Process each step (turn) in sequence
        for step_index, step in enumerate(a_gt_steps):
            # Execute all tool calls in this step
            for tool_index, tool_call in enumerate(step.a_gt):
                try:
                    # Process any placeholders in arguments using execution context
                    processed_arguments = self._process_placeholders(tool_call.arguments, execution_context)
                    
                    # Execute the tool with processed arguments
                    result = self.tool_manager.invoke_tool(tool_call.tool_name, processed_arguments)
                    
                    # Store result in execution context for future steps
                    execution_context[f"{tool_call.tool_name}.output"] = result
                    
                    # Record successful execution
                    executed_trajectory.append({
                        "step_index": step_index,
                        "tool_index": tool_index,
                        "tool_name": tool_call.tool_name,
                        "arguments": processed_arguments,
                        "status": "success",
                        "result": result
                    })
                    
                    # Add assistant message with tool call to conversation flow (will be done in generate_for_query)
                    # Add tool response message to conversation flow (will be done in generate_for_query)
                    
                except Exception as e:
                    overall_status = "error"
                    executed_trajectory.append({
                        "step_index": step_index,
                        "tool_index": tool_index,
                        "tool_name": tool_call.tool_name,
                        "arguments": tool_call.arguments,
                        "status": "error",
                        "error_message": str(e)
                    })
                    # Break on error as subsequent tools might depend on the failed one
                    break
            
            # If we encountered an error in this step, stop processing further steps
            if overall_status == "error":
                break
        
        # Add final assistant response using the expected outcome (will be done in generate_for_query)
        
        # Return verification result
        result = {
            "overall_status": overall_status, 
            "trajectory": executed_trajectory
        }
            
        return result

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

    def _review_blueprint_with_llm(self, q: str, a_gt_steps: List[aGTStep], o_gt: str, verification_result: Dict[str, Any]) -> Optional[str]:
        # Convert a_gt_steps to a JSON string for review
        a_gt_steps_str = json.dumps([{"a_gt": [call.model_dump() for call in step.a_gt]} for step in a_gt_steps], indent=2)
        verification_result_str = json.dumps(verification_result, indent=2)

        system_prompt = "You are an expert AI assistant. Your task is to review a generated plan (q, a_gt_steps, o_gt) and its execution trace."
        user_prompt = f'''
Original Query (q):
{q}

Generated Tool Call Steps (a_gt_steps):
{a_gt_steps_str}

Generated Expected Outcome (o_gt):
{o_gt}

Actual Execution Trace and Verification Result:
{verification_result_str}

Please review the following aspects:
1.  Clarity and relevance of 'q'.
2.  Logical coherence and correctness of 'a_gt_steps' to address 'q'.
3.  Appropriateness of multi-turn or parallel tool calls organization.
4.  Appropriateness and feasibility of tool calls given typical tool functionalities.
5.  Proper use of placeholder values where needed for dependent tool calls.
6.  Realism and achievability of 'o_gt'.
7.  Assessment of the execution trace: Did tools execute as expected? Were there errors?
8.  Alignment: Does the final state suggested by the execution trace align with 'o_gt'?
9.  Overall quality: Provide an overall assessment (e.g., Excellent, Good, Fair, Poor) and concise suggestions for improvement if any.

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
        
        # 1. Generate A_GT_STEPS and O_GT based on Q
        generation_result = self._generate_blueprint(q)
        if not generation_result:
            print(f"Failed to generate a_gt_steps and o_gt for query: {q}.")
            return None
            
        a_gt_steps, o_gt = generation_result
        
        # 2. Execute and Verify A_GT_STEPS - also store conversation messages
        conversation_messages: List[Dict[str, Any]] = [
            {
                "role": "system", 
                "content": "You are a helpful assistant that uses tools to fulfill user requests."
            },
            {
                "role": "user",
                "content": q
            }
        ] 
        # Execute tools and get verification result
        verification_result = self._execute_and_verify_a_gt_steps(q, a_gt_steps, o_gt)
        
        # Populate conversation messages for successful verifications
        if verification_result.get("overall_status") == "success":
            # Process for each turn in the conversation
            turn_index = 0
            execution_context = {}  # Store results from previous tool calls for placeholder substitution
            
            for step_index, step in enumerate(a_gt_steps):
                tool_calls = step.a_gt
                
                # Create assistant message with all tool calls in this step
                tool_call_structures = []
                for i, tool_call in enumerate(tool_calls):
                    # Prepare tool call with arguments
                    # If there are placeholders in arguments, substitute them from execution_context
                    arguments = self._process_placeholders(tool_call.arguments, execution_context)
                    
                    tool_call_structures.append({
                        "id": f"call_{turn_index}_{i}",
                        "type": "function",
                        "function": {
                            "name": tool_call.tool_name,
                            "arguments": json.dumps(arguments)
                        }
                    })
                
                # Add assistant message with tool calls
                assistant_message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_call_structures
                }
                conversation_messages.append(assistant_message)
                
                # Add tool response messages for each tool in this step
                for i, tool_call in enumerate(tool_calls):
                    tool_call_id = f"call_{turn_index}_{i}"
                    result = None
                    
                    # Find the result for this tool call from verification trajectory
                    for exec_step in verification_result.get("trajectory", []):
                        if (exec_step.get("step_index") == step_index and 
                            exec_step.get("tool_index") == i and 
                            exec_step.get("status") == "success"):
                            result = exec_step.get("result")
                            # Store result in execution context for potential future placeholder substitution
                            execution_context[f"{tool_call.tool_name}.output"] = result
                            break
                    
                    # Add tool response message
                    if result is not None:
                        conversation_messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "name": tool_call.tool_name,
                            "content": result
                        })
                
                turn_index += 1
            
            # Add final assistant response
            conversation_messages.append({
                "role": "assistant",
                "content": o_gt
            })
        
        # 3. (Optional) Review with LLM
        llm_review = None
        if with_review:
            try:
                llm_review = self._review_blueprint_with_llm(q, a_gt_steps, o_gt, verification_result)
            except Exception as e:
                print(f"Error during LLM review for query: {q}: {e}")
                # Continue without review if it fails

        # Prepare data for GeneratedData object
        data_kwargs = {
            "q": q,
            "a_gt_steps": a_gt_steps,
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

    def _process_placeholders(self, arguments: Dict[str, Any], execution_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process placeholders in arguments using execution context from previous steps.
        
        Args:
            arguments: Dictionary of arguments that may contain placeholders
            execution_context: Dictionary with results from previous tool executions
            
        Returns:
            Dictionary with processed arguments with placeholders replaced where possible
        """
        processed_args = {}
        
        for arg_name, arg_value in arguments.items():
            if isinstance(arg_value, str) and "{{" in arg_value and "}}" in arg_value:
                # This is a placeholder, try to substitute it
                placeholder_pattern = r"{{([^{}]+)}}"
                match = re.search(placeholder_pattern, arg_value)
                if match:
                    placeholder_key = match.group(1)
                    # Check if the placeholder key is in execution_context
                    if placeholder_key in execution_context:
                        # Replace the placeholder with the actual value
                        processed_args[arg_name] = execution_context[placeholder_key]
                    else:
                        # Keep the placeholder as is if we can't resolve it
                        print(f"Warning: Could not resolve placeholder '{placeholder_key}' in argument '{arg_name}'")
                        processed_args[arg_name] = arg_value
                else:
                    processed_args[arg_name] = arg_value
            else:
                processed_args[arg_name] = arg_value
        
        return processed_args


if __name__ == "__main__":
    print("Starting APIGenMTGenerator with actual components...")

    try:
        llm_client = LLMClient() 
        print("LLMClient initialized.")

        tool_manager = ToolManager(
            llm=llm_client,
        )
        print("ToolManager initialized.")

        generator = APIGenMTGenerator(
            llm=llm_client,
            tool_manager=tool_manager
        )
        print("APIGenMTGenerator initialized with actual components.")

        print("\n=== Example: Generate data for a single query with multi-turn steps ===")
        # Example with multi-turn interaction
        multi_turn_query = "Search for the ADEX 2025 event date, then register the event schedule in the calendar"
        single_result = generator.generate_for_query(multi_turn_query)
        
        if single_result:
            print("\n--- Multi-turn Query Result ---")
            # Print a more human-readable version of the result
            print("Query:", single_result.q)
            print("\nGround Truth Steps:")
            for i, step in enumerate(single_result.a_gt_steps):
                print(f"  Step {i+1}:")
                for j, tool in enumerate(step.a_gt):
                    print(f"    Tool {j+1}: {tool.tool_name}")
                    print(f"      Arguments: {json.dumps(tool.arguments, ensure_ascii=False, indent=8)}")
            
            print("\nExpected Outcome:", single_result.o_gt)
            print("\nVerification Result Status:", single_result.verification_result.get("overall_status", "unknown"))
            
            # Print full JSON if needed
            print("\nFull JSON Result Available")
            print(single_result.model_dump_json(indent=2))
        else:
            print("\nFailed to generate data for the multi-turn query.") 

    except Exception as e:
        print(f"An error occurred during execution with actual components: {e}")
        import traceback
        traceback.print_exc()

    print("\nExample finished.")
