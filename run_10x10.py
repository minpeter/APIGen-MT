#!/usr/bin/env python3
import json, os, sys, random, signal
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from llm_client import LocalOpenAILLMClient
from tool_manager import ToolManager
from apigen_step_by_step import StepByStepGenerator

output_path = 'data/generated/stateful_10x10_datapoints.jsonl'
log_path = 'data/generated/stateful_10x10_log.txt'

def log(msg):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(log_path, 'a') as f:
        f.write(line + '\n')

log("Starting generation: 10 datapoints x 10 actions")

api_key = os.getenv("OPENAI_API_KEY")
api_base = os.getenv("OPENAI_API_BASE")
if not api_key or not api_base:
    log("ERROR: API credentials not set")
    sys.exit(1)

log(f"API base: {api_base}")

llm_client = LocalOpenAILLMClient(url=api_base, api_key=api_key, api_model='z-ai/glm-5.1', hf_tokenizer_id=None)

tool_manager = ToolManager(
    llm=llm_client,
    tool_pool_path='magnet_tool_extraction/bfcl_v3_tools_with_outputs.jsonl',
    invocation_examples_path='magnet_tool_extraction/bfcl_v3_invocation_examples.jsonl'
)
log(f"Loaded {len(tool_manager.python_tool_instances)} tool classes, {len(tool_manager.api_name_to_class_key)} API mappings")

generator = StepByStepGenerator(llm_client=llm_client, tool_manager=tool_manager, num_actions=10)

categories = list(set(t.get('category', 'Unknown') for t in tool_manager.tool_schemas))
log(f"Categories: {categories}")

# Remove old output
if os.path.exists(output_path):
    os.remove(output_path)

datapoints = []
for i in range(10):
    focus = random.choice(categories)
    log(f"=== Datapoint {i+1}/10 | Focus: {focus} ===")
    dp = generator.generate_datapoint(focus_category=focus)
    if dp:
        dp_dict = dp.model_dump()
        dp_dict['timestamp'] = datetime.now().isoformat()
        dp_dict['generation_attempt'] = i
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(dp_dict, ensure_ascii=False) + '\n')
        datapoints.append(dp)
        
        init_keys = list(dp.initial_api_state.keys()) if dp.initial_api_state else []
        log(f"✓ Datapoint {i+1} saved | Tools: {dp.trajectory.tools_used} | State keys: {init_keys}")
    else:
        log(f"✗ Datapoint {i+1} FAILED, retrying...")
        # Retry once
        focus2 = random.choice(categories)
        dp2 = generator.generate_datapoint(focus_category=focus2)
        if dp2:
            dp2_dict = dp2.model_dump()
            dp2_dict['timestamp'] = datetime.now().isoformat()
            dp2_dict['generation_attempt'] = i
            with open(output_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(dp2_dict, ensure_ascii=False) + '\n')
            datapoints.append(dp2)
            init_keys = list(dp2.initial_api_state.keys()) if dp2.initial_api_state else []
            log(f"✓ Datapoint {i+1} saved (retry) | Tools: {dp2.trajectory.tools_used} | State keys: {init_keys}")
        else:
            log(f"✗ Datapoint {i+1} FAILED permanently")

log(f"=== DONE: {len(datapoints)}/10 datapoints generated ===")
if datapoints:
    import os as _os
    size = _os.path.getsize(output_path)
    log(f"Output: {output_path} ({size/1024:.0f} KB)")
