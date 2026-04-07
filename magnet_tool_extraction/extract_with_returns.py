#!/usr/bin/env python3
"""
Extract tool invocations WITH simulated return values.

BFCL_v3 doesn't include actual return values, but we can:
1. Use initial_config to understand what state the tools operate on
2. For stateful tools (like file system), simulate returns
3. Document what the expected behavior would be
"""

import json
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict


def simulate_tool_return(function_name: str, arguments: Dict, initial_config: Dict) -> Dict:
    """
    Simulate what a tool would return based on its function and initial state.
    
    This is a simplified simulation - real execution would require actual tool implementations.
    """
    
    # File system operations
    if function_name in ['ls', 'cat', 'pwd']:
        return simulate_filesystem_return(function_name, arguments, initial_config)
    
    # Twitter/Social media operations
    elif function_name in ['post_tweet', 'get_timeline', 'like_tweet']:
        return simulate_twitter_return(function_name, arguments, initial_config)
    
    # Stock/trading operations
    elif function_name in ['get_stock_info', 'buy_stock', 'sell_stock']:
        return simulate_trading_return(function_name, arguments, initial_config)
    
    # Vehicle operations
    elif function_name in ['startEngine', 'lockDoors', 'check_tire_pressure']:
        return simulate_vehicle_return(function_name, arguments, initial_config)
    
    # Generic operations
    else:
        return {
            'status': 'success',
            'message': f'Function {function_name} executed successfully',
            'note': 'Actual return value would depend on implementation',
            'arguments_used': arguments
        }


def simulate_filesystem_return(function_name: str, arguments: Dict, config: Dict) -> Dict:
    """Simulate file system tool returns."""
    
    fs_config = config.get('GorillaFileSystem', {})
    
    if function_name == 'ls':
        # List directory contents
        return {
            'status': 'success',
            'result': {
                'type': 'list',
                'contents': ['file1.txt', 'file2.pdf', 'subdir/'],
                'note': 'Directory listing would show actual files from initial_config'
            },
            'simulated': True
        }
    
    elif function_name == 'cat':
        # Display file contents
        return {
            'status': 'success',
            'result': {
                'type': 'string',
                'content': 'File content would be retrieved from initial_config',
                'file': arguments.get('file_name')
            },
            'simulated': True
        }
    
    elif function_name == 'pwd':
        return {
            'status': 'success',
            'result': {
                'type': 'string',
                'current_directory': '/workspace/document',
                'note': 'Would show actual path from traversal history'
            },
            'simulated': True
        }
    
    elif function_name == 'cd':
        return {
            'status': 'success',
            'result': {
                'type': 'acknowledgment',
                'message': f"Changed directory to '{arguments.get('folder')}'",
                'new_path': f"/workspace/{arguments.get('folder')}"
            },
            'simulated': True
        }
    
    elif function_name == 'mkdir':
        return {
            'status': 'success',
            'result': {
                'type': 'acknowledgment',
                'message': f"Created directory '{arguments.get('dir_name')}'"
            },
            'simulated': True
        }
    
    elif function_name == 'mv':
        return {
            'status': 'success',
            'result': {
                'type': 'acknowledgment',
                'message': f"Moved '{arguments.get('source')}' to '{arguments.get('destination')}'"
            },
            'simulated': True
        }
    
    return {'status': 'unknown_function', 'function': function_name}


def simulate_twitter_return(function_name: str, arguments: Dict, config: Dict) -> Dict:
    """Simulate Twitter API tool returns."""
    
    twitter_config = config.get('TwitterAPI', {})
    
    if function_name == 'get_tweet':
        return {
            'status': 'success',
            'result': {
                'type': 'dict',
                'tweet': {
                    'id': arguments.get('tweet_id', '0'),
                    'username': 'analyst_pro',
                    'content': 'Just finished analyzing the reports!',
                    'likes': 42,
                    'retweets': 12
                },
                'note': 'Actual tweet data would come from initial_config'
            },
            'simulated': True
        }
    
    elif function_name == 'post_tweet':
        return {
            'status': 'success',
            'result': {
                'type': 'dict',
                'tweet_id': '123',
                'message': 'Tweet posted successfully',
                'content': arguments.get('text_content', '')[:50]
            },
            'simulated': True
        }
    
    return {'status': 'unknown_function', 'function': function_name}


def simulate_trading_return(function_name: str, arguments: Dict, config: Dict) -> Dict:
    """Simulate trading/stock API returns."""
    
    if function_name == 'get_stock_info':
        return {
            'status': 'success',
            'result': {
                'type': 'dict',
                'stock': {
                    'symbol': arguments.get('symbol'),
                    'price': 142.50,
                    'change': +2.35,
                    'change_percent': '+1.67%',
                    'volume': 12345678,
                    'market_cap': '3.5T'
                },
                'note': 'Actual stock data would require real API call'
            },
            'simulated': True
        }
    
    elif function_name == 'buy_stock':
        return {
            'status': 'success',
            'result': {
                'type': 'dict',
                'order_id': 'ORD-12345',
                'message': f"Bought {arguments.get('quantity')} shares of {arguments.get('symbol')}",
                'total_cost': 1425.00
            },
            'simulated': True
        }
    
    return {'status': 'unknown_function', 'function': function_name}


def simulate_vehicle_return(function_name: str, arguments: Dict, config: Dict) -> Dict:
    """Simulate vehicle control returns."""
    
    if function_name == 'startEngine':
        return {
            'status': 'success',
            'result': {
                'type': 'dict',
                'engine_status': 'running',
                'ignition_mode': arguments.get('ignitionMode', 'START'),
                'message': 'Engine started successfully'
            },
            'simulated': True
        }
    
    elif function_name == 'lockDoors':
        doors = arguments.get('door', ['all'])
        unlock = arguments.get('unlock', False)
        action = 'unlocked' if unlock else 'locked'
        
        return {
            'status': 'success',
            'result': {
                'type': 'dict',
                'door_status': {door: action for door in doors},
                'message': f"Doors {action}: {', '.join(doors)}"
            },
            'simulated': True
        }
    
    elif function_name == 'check_tire_pressure':
        return {
            'status': 'success',
            'result': {
                'type': 'dict',
                'tire_pressure': {
                    'front_left': 32.5,
                    'front_right': 32.3,
                    'rear_left': 31.8,
                    'rear_right': 32.0
                },
                'unit': 'PSI',
                'status': 'normal'
            },
            'simulated': True
        }
    
    return {'status': 'unknown_function', 'function': function_name}


def extract_with_returns():
    """
    Extract invocation examples with simulated return values.
    """
    
    data_dir = Path("/home/ishalyminov/data/magnet_mt/data/BFCL_v3")
    
    # Multi-turn test files
    test_files = [
        "BFCL_v3_multi_turn_base.json",
        "BFCL_v3_multi_turn_composite.json",
        "BFCL_v3_multi_turn_long_context.json",
        "BFCL_v3_multi_turn_miss_func.json",
        "BFCL_v3_multi_turn_miss_param.json",
    ]
    
    # Corresponding answer files
    answer_files = [
        f"possible_answer/{f}" for f in test_files
    ]
    
    results = []
    
    for test_file, answer_file in zip(test_files, answer_files):
        test_path = data_dir / test_file
        answer_path = data_dir / answer_file
        
        if not test_path.exists():
            print(f"⚠️  Skipping {test_file} - not found")
            continue
        
        print(f"📄 Processing {test_file}...")
        
        # Load both test and answer
        with open(test_path, 'r') as f:
            test_lines = f.readlines()
        
        with open(answer_path, 'r') as f:
            answer_lines = f.readlines()
        
        # Process each test case
        for test_line, answer_line in zip(test_lines, answer_lines):
            if not test_line.strip():
                continue
            
            try:
                test_data = json.loads(test_line)
                answer_data = json.loads(answer_line)
                
                test_id = test_data['id']
                initial_config = test_data.get('initial_config', {})
                ground_truth = answer_data.get('ground_truth', [])
                
                # Extract each turn
                for turn_idx, turn_calls in enumerate(ground_truth):
                    if isinstance(turn_calls, list):
                        for call in turn_calls:
                            # Parse the call string
                            # Format: function_name(arg1=val1, arg2=val2)
                            import re
                            match = re.match(r'(\w+)\((.*)\)', call)
                            
                            if match:
                                func_name = match.group(1)
                                args_str = match.group(2)
                                
                                # Simple argument parsing
                                arguments = {}
                                if args_str:
                                    # Parse key=value pairs
                                    for arg in args_str.split(','):
                                        if '=' in arg:
                                            k, v = arg.split('=', 1)
                                            k = k.strip()
                                            v = v.strip().strip("'\"")
                                            # Try to convert to appropriate type
                                            try:
                                                v = json.loads(v)
                                            except:
                                                pass
                                            arguments[k] = v
                                
                                # Simulate return value
                                simulated_return = simulate_tool_return(
                                    func_name, 
                                    arguments, 
                                    initial_config
                                )
                                
                                result = {
                                    'id': f"{test_id}_turn{turn_idx}_{func_name}",
                                    'test_case_id': test_id,
                                    'turn_index': turn_idx,
                                    'function_name': func_name,
                                    'arguments': arguments,
                                    'call_string': call,
                                    'initial_config_summary': {
                                        'tools': list(initial_config.keys()),
                                        'has_state': bool(initial_config)
                                    },
                                    'simulated_return': simulated_return,
                                    'note': 'Return value is simulated based on initial_config state'
                                }
                                
                                results.append(result)
            
            except Exception as e:
                print(f"  ⚠️  Error processing test case: {e}")
                continue
    
    return results


def main():
    """Main function."""
    
    print("=" * 80)
    print("EXTRACTING TOOL INVOCATIONS WITH SIMULATED RETURNS")
    print("=" * 80)
    
    results = extract_with_returns()
    
    print(f"\n✅ Extracted {len(results)} invocations with simulated returns")
    
    # Save to file
    output_path = "bfcl_v3_invocations_with_returns.jsonl"
    print(f"\n💾 Saving to {output_path}...")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"✅ Saved {len(results)} invocations")
    
    # Show samples
    print("\n" + "=" * 80)
    print("SAMPLE INVOCATIONS WITH RETURNS")
    print("=" * 80)
    
    for i, result in enumerate(results[:5], 1):
        print(f"\n{i}. {result['function_name']}")
        print(f"   Arguments: {result['arguments']}")
        print(f"   Call: {result['call_string']}")
        print(f"   Return: {json.dumps(result['simulated_return'], indent=6)}")


if __name__ == "__main__":
    main()