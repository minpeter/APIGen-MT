"""
LLM-based predictor for tool output type and description.
Uses LLM-as-a-judge to predict what a tool returns based on its schema and invocation contexts.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from openai import OpenAI

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_client import LLMClient, LocalOpenAILLMClient
from runtime_config import DEFAULT_API_BASE, DEFAULT_MODEL


class OutputPrediction(BaseModel):
    """Schema for LLM output prediction"""
    output_type: str
    output_description: str


class LLMOutputPredictor:
    """
    Uses LLM to predict output type and description for tools based on:
    - Tool schema (name, description, parameters)
    - Invocation contexts (how the tool is used in practice)

    The client uses the repository's OpenAI-compatible runtime configuration.
    """

    def __init__(self, client_type: str = "openai-compatible", debug: bool = False):
        """
        Initialize the LLM output predictor.

        Args:
            client_type: Retained for backwards compatibility.
            debug: Enable debug logging

        The extraction workflow is still opt-in and requires an API key.
        """
        self.debug = debug

        # Use the shared OpenAI-compatible runtime defaults.
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_API_BASE", DEFAULT_API_BASE)

        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = DEFAULT_MODEL
        self.client_type = "openai-compatible"

    def predict_output(
        self,
        tool_schema: Dict[str, Any],
        invocation_contexts: List[Dict[str, Any]],
        max_contexts: int = 5,
        max_retries: int = 3
    ) -> OutputPrediction:
        """
        Predict output type and description for a tool.

        Args:
            tool_schema: Tool definition schema
            invocation_contexts: List of invocation contexts
            max_contexts: Maximum number of contexts to include
            max_retries: Maximum number of retry attempts on LLM failure (default: 3)

        Returns:
            OutputPrediction with output_type and output_description
        """
        # Build the prompt
        system_prompt = self._get_system_prompt()
        prompt = self._build_prompt(tool_schema, invocation_contexts[:max_contexts])

        if self.debug:
            print(f"\n{'='*80}")
            print(f"Predicting output for: {tool_schema.get('api_name', 'unknown')}")
            print(f"{'='*80}")
            print(f"System prompt:\n{system_prompt}")
            print(f"\nUser prompt:\n{prompt}")
            print(f"{'='*80}\n")

        # Retry loop
        for attempt in range(1, max_retries + 1):
            try:
                if self.client_type == "openai-compatible":
                    # Use OpenAI client directly
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,
                        max_tokens=500
                    )

                    content = response.choices[0].message.content

                    if self.debug:
                        print(f"LLM Response (attempt {attempt}): {content}")

                    # Parse JSON from response
                    # Try to extract JSON from the response
                    import re
                    json_match = re.search(r'\{[^{}]*"output_type"[^{}]*"output_description"[^{}]*\}', content, re.DOTALL)
                    if json_match:
                        prediction_dict = json.loads(json_match.group())
                    else:
                        # Try parsing the whole response
                        prediction_dict = json.loads(content)

                    # Add 5-second timeout after successful LLM call
                    time.sleep(5)
                    return OutputPrediction(**prediction_dict)
                else:
                    # Use existing LLM client
                    prediction, reasoning = self.llm.json_output(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        schema=OutputPrediction,
                        reasoning=True
                    )

                    if self.debug:
                        print(f"Reasoning: {reasoning}")
                        print(f"Prediction: {prediction}")

                    # Add 5-second timeout after successful LLM call
                    time.sleep(5)
                    return OutputPrediction(**prediction)

            except Exception as e:
                if attempt < max_retries:
                    print(f"Attempt {attempt}/{max_retries} failed for {tool_schema.get('api_name', 'unknown')}: {e}")
                    print(f"Retrying...")
                else:
                    print(f"Error predicting output after {max_retries} attempts: {e}")
                    # Return default values
                    return OutputPrediction(
                        output_type="unknown",
                        output_description="Failed to predict output description"
                    )

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the LLM"""
        return """You are an expert at analyzing function/tool schemas and predicting what they return.

Given information about a function/tool:
1. Its name and description
2. Its parameters (arguments it accepts)
3. Example invocations showing how it's used in practice

Your task is to predict:
1. **output_type**: The type of data the function returns (e.g., "string", "integer", "boolean", "dict", "list", "file content", "API response", "operation status", etc.)
2. **output_description**: A clear description of what the function returns, including the structure if it's a complex type

Guidelines:
- Be specific about the output type (e.g., "weather data dict" instead of just "dict")
- Include important fields if returning a structured type
- Mention if the function returns success/failure status
- Consider what makes sense given the function's purpose and parameters
- Use the invocation contexts to understand real-world usage patterns

Respond ONLY with a valid JSON object matching the schema:
{
  "output_type": "string",
  "output_description": "string"
}"""

    def _build_prompt(
        self,
        tool_schema: Dict[str, Any],
        invocation_contexts: List[Dict[str, Any]]
    ) -> str:
        """Build the prompt for the LLM"""

        # Extract tool information
        tool_name = tool_schema.get('tool_name', 'unknown')
        api_name = tool_schema.get('api_name', 'unknown')
        api_description = tool_schema.get('api_description', 'No description available')
        parameters = tool_schema.get('parameters', {})

        # Build the prompt
        prompt_parts = [
            f"# Function Information",
            f"",
            f"**Tool Name**: {tool_name}",
            f"**API Name**: {api_name}",
            f"**Description**: {api_description}",
            f"",
        ]

        # Add parameters
        if parameters:
            prompt_parts.append("## Parameters")
            prompt_parts.append("")
            prompt_parts.append(f"```json")
            # Convert to dict if it's a Pydantic model or has to_dict method
            if hasattr(parameters, 'to_dict'):
                parameters = parameters.to_dict()
            elif hasattr(parameters, 'model_dump'):
                parameters = parameters.model_dump()
            prompt_parts.append(json.dumps(parameters, indent=2))
            prompt_parts.append(f"```")
            prompt_parts.append("")

        # Add invocation contexts
        if invocation_contexts:
            prompt_parts.append("## Example Invocations")
            prompt_parts.append("")

            for i, ctx in enumerate(invocation_contexts, 1):
                user_message = ctx.get('user_message', 'N/A')
                assistant_message = ctx.get('assistant_message', 'N/A')
                tool_calls = ctx.get('tool_calls', [])

                prompt_parts.append(f"### Context {i}")
                prompt_parts.append(f"**User**: {user_message}")
                prompt_parts.append(f"**Assistant**: {assistant_message}")

                if tool_calls:
                    prompt_parts.append(f"**Tool Calls**:")
                    for tc in tool_calls:
                        prompt_parts.append(f"- {tc.get('name', 'unknown')}: {json.dumps(tc.get('arguments', {}))}")

                prompt_parts.append("")

        # Add the task
        prompt_parts.extend([
            "## Task",
            f"",
            f"Based on the function information and example invocations above, predict:",
            f"1. The output type this function returns",
            f"2. A clear description of what the output contains",
            f"",
            f"Respond with a JSON object containing 'output_type' and 'output_description'.",
        ])

        return "\n".join(prompt_parts)


def predict_outputs_for_tools(
    tools: List[Dict[str, Any]],
    invocations: List[Dict[str, Any]],
    client_type: str = "openai-compatible",
    max_contexts: int = 5,
    debug: bool = False
) -> List[Dict[str, Any]]:
    """
    Predict output types and descriptions for multiple tools.

    Args:
        tools: List of tool definitions
        invocations: List of all invocation examples
        client_type: Retained for backwards compatibility.
        max_contexts: Maximum invocation contexts per tool
        debug: Enable debug mode

    Returns:
        List of tool definitions with added output_type and output_description
    """
    predictor = LLMOutputPredictor(client_type=client_type, debug=debug)

    # Index invocations by tool name
    invocations_by_tool = {}
    for inv in invocations:
        tool_name = inv.get('tool_name')
        if tool_name not in invocations_by_tool:
            invocations_by_tool[tool_name] = []
        invocations_by_tool[tool_name].append(inv)

    enhanced_tools = []

    print(f"\n{'='*80}")
    print("🤖 PREDICTING OUTPUTS USING LLM")
    print(f"{'='*80}")
    print(f"Max contexts per tool: {max_contexts}")
    print(f"Debug mode: {debug}")
    print()

    for i, tool in enumerate(tools, 1):
        tool_name = tool.get('tool_name')
        api_name = tool.get('api_name', 'unknown')

        print(f"Processing tool {i}/{len(tools)}: {api_name}")

        # Get invocation contexts for this tool
        tool_invocations = invocations_by_tool.get(tool_name, [])

        # Predict output
        prediction = predictor.predict_output(tool, tool_invocations, max_contexts)

        # Create enhanced tool definition
        enhanced_tool = {
            **tool,
            'output_type': prediction.output_type,
            'output_description': prediction.output_description
        }

        enhanced_tools.append(enhanced_tool)

    return enhanced_tools


if __name__ == "__main__":
    # Test the predictor
    from dotenv import load_dotenv
    load_dotenv()

    test_tool = {
        "tool_name": "weather_api",
        "api_name": "get_weather",
        "api_description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name or zip code"
                }
            },
            "required": ["location"]
        }
    }

    test_contexts = [
        {
            "user_message": "What's the weather in San Francisco?",
            "assistant_message": "Let me check the weather for you.",
            "tool_calls": [
                {
                    "name": "get_weather",
                    "arguments": {"location": "San Francisco"}
                }
            ]
        }
    ]

    predictor = LLMOutputPredictor(client_type="openai-compatible", debug=True)
    prediction = predictor.predict_output(test_tool, test_contexts)

    print(f"\n{'='*80}")
    print("PREDICTION RESULT")
    print(f"{'='*80}")
    print(f"Output Type: {prediction.output_type}")
    print(f"Output Description: {prediction.output_description}")
