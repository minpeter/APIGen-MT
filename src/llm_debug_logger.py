"""
Debug logger for LLM calls.
Provides detailed logging of prompts, responses, and parsing.
"""

import json
from datetime import datetime


def log_llm_call(debug_mode: bool, call_type: str, model: str, endpoint: str, 
                 messages: list, kwargs: dict, response: str = None, 
                 reasoning: str = None, parsed_output: dict = None):
    """
    Log LLM call details if debug mode is enabled.
    
    Args:
        debug_mode: Whether to log
        call_type: Type of call (chat, completions, json_output)
        model: Model name
        endpoint: API endpoint
        messages: Messages sent to LLM
        kwargs: Additional parameters
        response: Raw response from LLM (optional)
        reasoning: Extracted reasoning (optional)
        parsed_output: Parsed JSON output (optional)
    """
    if not debug_mode:
        return
    
    print("\n" + "=" * 80)
    print(f"📤 LLM API CALL - {call_type}()")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Model: {model}")
    print(f"Endpoint: {endpoint}")
    
    # Print messages
    print(f"\n📝 Messages ({len(messages)} total):")
    for idx, msg in enumerate(messages):
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        
        # Truncate long content
        if len(content) > 500:
            display_content = content[:500] + f"\n... (truncated, {len(content)} total chars)"
        else:
            display_content = content
        
        print(f"\n[{idx}] Role: {role}")
        print(f"    Content:")
        for line in display_content.split('\n'):
            print(f"      {line}")
    
    # Print kwargs
    if kwargs:
        print(f"\n📋 kwargs:")
        print(json.dumps(kwargs, indent=2, ensure_ascii=False))
    
    # Print response if available
    if response is not None:
        print("\n" + "=" * 80)
        print("📥 LLM RAW RESPONSE")
        print("=" * 80)
        
        # Truncate long responses
        if len(response) > 1000:
            display_response = response[:1000] + f"\n... (truncated, {len(response)} total chars)"
        else:
            display_response = response
        
        for line in display_response.split('\n'):
            print(f"  {line}")
    
    # Print reasoning if available
    if reasoning:
        print("\n" + "=" * 80)
        print("🤔 EXTRACTED REASONING")
        print("=" * 80)
        for line in reasoning.split('\n'):
            print(f"  {line}")
    
    # Print parsed output if available
    if parsed_output is not None:
        print("\n" + "=" * 80)
        print("✅ PARSED OUTPUT")
        print("=" * 80)
        print(json.dumps(parsed_output, indent=2, ensure_ascii=False))
    
    print("\n" + "=" * 80 + "\n")