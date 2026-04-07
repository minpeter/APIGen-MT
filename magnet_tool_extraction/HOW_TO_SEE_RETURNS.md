# How to See What Tool Calls Return

## 📋 Overview

**Important**: BFCL_v3 (Berkeley Function Calling Leaderboard) is designed to test **function calling**, not function execution. Therefore:

- ✅ **Has**: Function definitions, call signatures, input arguments
- ✅ **Has**: Initial state/configuration for stateful tools
- ❌ **Does NOT have**: Actual return values from executed functions

---

## 🎯 What's Available in BFCL_v3

### 1. **Function Definitions**
Located in test files and `multi_turn_func_doc/`:
- Function names
- Parameter schemas
- Return type expectations
- Descriptions

### 2. **Ground Truth Calls**
Located in `possible_answer/` directory:
- Exact function call syntax
- Input arguments
- Call order in multi-turn conversations

### 3. **Initial Configuration**
Located in test files:
- Simulated state for stateful tools
- Mock databases (e.g., file system structure, Twitter posts)
- Pre-populated data

### 4. **Execution Metadata** (for exec tests)
Located in `BFCL_v3_exec_*.json`:
- `execution_result_type`: Expected return type
- Function implementations (some tests have actual code)

---

## 💡 How to See Returns

You have **3 options**:

### Option 1: Simulated Returns (What We Extracted)

We created simulated returns based on:
- Function type
- Input arguments
- Initial state

**File**: `bfcl_v3_invocations_with_returns.jsonl`

**Example**:
```json
{
  "function_name": "get_stock_info",
  "arguments": {"symbol": "NVDA"},
  "call_string": "get_stock_info(symbol='NVDA')",
  "simulated_return": {
    "status": "success",
    "result": {
      "type": "dict",
      "stock": {
        "symbol": "NVDA",
        "price": 142.50,
        "change": +2.35,
        "change_percent": "+1.67%",
        "volume": 12345678
      },
      "note": "Actual stock data would require real API call"
    },
    "simulated": true
  }
}
```

---

### Option 2: Use Initial Configuration

The `initial_config` field contains the mock state that tools operate on.

**Example**: File system state
```json
{
  "GorillaFileSystem": {
    "root": {
      "workspace": {
        "type": "directory",
        "contents": {
          "document": {
            "type": "directory",
            "contents": {
              "final_report.pdf": {
                "type": "file",
                "content": "Year2024 This is the final report..."
              }
            }
          }
        }
      }
    }
  }
}
```

You can trace what operations would return by simulating against this state.

---

### Option 3: Implement Actual Tools

For real execution, you would need to:
1. Implement the actual tool functions
2. Use the initial_config as initial state
3. Execute the function calls
4. Capture actual returns

---

## 📊 Examples by Category

### 1. File System Operations

**Function**: `cd` (259 calls)
```python
# Input
cd(folder='document')

# Simulated Return
{
  "status": "success",
  "result": {
    "type": "acknowledgment",
    "message": "Changed directory to 'document'",
    "new_path": "/workspace/document"
  }
}
```

**Function**: `ls`
```python
# Input
ls()

# Simulated Return
{
  "status": "success",
  "result": {
    "type": "list",
    "contents": ["file1.txt", "file2.pdf", "subdir/"]
  }
}
```

**Function**: `cat`
```python
# Input
cat(file_name='final_report.pdf')

# Simulated Return
{
  "status": "success",
  "result": {
    "type": "string",
    "content": "File content from initial_config..."
  }
}
```

---

### 2. Stock/Finance Operations

**Function**: `get_stock_info` (215 calls)
```python
# Input
get_stock_info(symbol='NVDA')

# Simulated Return
{
  "status": "success",
  "result": {
    "type": "dict",
    "stock": {
      "symbol": "NVDA",
      "price": 142.50,
      "change": +2.35,
      "change_percent": "+1.67%",
      "volume": 12345678,
      "market_cap": "3.5T"
    }
  }
}
```

---

### 3. Vehicle Control Operations

**Function**: `lockDoors` (210 calls)
```python
# Input
lockDoors(unlock=True, door=['driver', 'passenger'])

# Simulated Return
{
  "status": "success",
  "result": {
    "type": "dict",
    "door_status": {
      "driver": "unlocked",
      "passenger": "unlocked"
    },
    "message": "Doors unlocked: driver, passenger"
  }
}
```

**Function**: `startEngine` (220 calls)
```python
# Input
startEngine(ignitionMode='START')

# Simulated Return
{
  "status": "success",
  "result": {
    "type": "dict",
    "engine_status": "running",
    "ignition_mode": "START",
    "message": "Engine started successfully"
  }
}
```

---

### 4. Social Media Operations

**Function**: `post_tweet` (171 calls)
```python
# Input
post_tweet(
    content='Managed to archive important data files!',
    tags=['#DataManagement', '#Efficiency']
)

# Simulated Return
{
  "status": "success",
  "result": {
    "type": "dict",
    "tweet_id": "123",
    "message": "Tweet posted successfully",
    "content": "Managed to archive important data files!"
  }
}
```

---

### 5. Travel Booking Operations

**Function**: `book_flight` (205 calls)
```python
# Input
book_flight(
    access_token='abc123xyz',
    card_id='144756014165',
    travel_date='2024-11-10',
    travel_from='SFO',
    travel_to='LAX',
    travel_class='business',
    travel_cost=400.0
)

# Simulated Return
{
  "status": "success",
  "result": {
    "type": "dict",
    "booking_id": "BK-12345",
    "message": "Flight booked successfully",
    "details": {
      "date": "2024-11-10",
      "route": "SFO → LAX",
      "class": "business",
      "cost": 400.0
    }
  }
}
```

---

## 🔍 Accessing the Data

### Python Example

```python
import json

# Load invocations with simulated returns
with open('bfcl_v3_invocations_with_returns.jsonl', 'r') as f:
    invocations = [json.loads(line) for line in f]

# View a specific invocation
inv = invocations[0]
print(f"Function: {inv['function_name']}")
print(f"Arguments: {inv['arguments']}")
print(f"Return: {json.dumps(inv['simulated_return'], indent=2)}")
```

### Browse by Category

```python
from collections import defaultdict

# Group by function
by_function = defaultdict(list)
for inv in invocations:
    by_function[inv['function_name']].append(inv)

# Get all calls for a specific function
stock_calls = by_function['get_stock_info']
for call in stock_calls[:5]:
    print(f"Input: {call['arguments']}")
    print(f"Return: {call['simulated_return']}")
```

---

## 📈 Statistics

### Total Data Extracted
- **5,959 invocations** with simulated returns
- **86 unique functions**
- **Multi-turn conversations** preserved

### Top 10 Most Called Functions

1. **cd** - 259 calls (Storage)
2. **startEngine** - 220 calls (Vehicle Control)
3. **get_stock_info** - 215 calls (Finance)
4. **lockDoors** - 210 calls (Vehicle Control)
5. **book_flight** - 205 calls (Travel Booking)
6. **get_zipcode_based_on_city** - 184 calls (Location)
7. **get_flight_cost** - 180 calls (Travel Booking)
8. **pressBrakePedal** - 176 calls (Vehicle Control)
9. **post_tweet** - 171 calls (Social Media)
10. **get_order_details** - 161 calls (E-commerce)

---

## 🎯 Key Points

### ✅ What You Can See
- Function signatures and definitions
- Input arguments for all calls
- Simulated return values (estimated)
- Initial state/configuration
- Call order and context

### ⚠️ Limitations
- **Returns are simulated**, not actual execution results
- **BFCL_v3 is for function calling**, not execution testing
- **Real returns** would require:
  - Actual tool implementations
  - Real API connections
  - Execution environment

### 💡 Best Practices
1. Use simulated returns for understanding tool behavior
2. Reference initial_config for state-dependent operations
3. Implement actual tools if you need real execution
4. Combine with function definitions for complete picture

---

## 📁 Related Files

| File | Description |
|------|-------------|
| `bfcl_v3_invocations_with_returns.jsonl` | Invocations with simulated returns |
| `bfcl_v3_all_tool_definitions.jsonl` | All tool definitions |
| `bfcl_v3_invocation_examples.jsonl` | Original invocation examples |
| `extract_with_returns.py` | Extraction script |
| `tool_analysis.ipynb` | Jupyter notebook analysis |

---

## 🚀 Next Steps

1. **Browse the simulated returns**: Open `bfcl_v3_invocations_with_returns.jsonl`
2. **Analyze patterns**: Use the Jupyter notebook `tool_analysis.ipynb`
3. **Implement real tools**: If you need actual execution
4. **Combine datasets**: Use definitions + calls + returns together

---

**Generated**: 2026-04-05  
**Data**: BFCL_v3 (Berkeley Function Calling Leaderboard)  
**Note**: Return values are simulated based on function type and initial configuration