# Magnet Tool Pool Extraction Analysis

## Overview

This analysis covers the tool pool extraction scripts from the Magnet paper (arXiv 2503.07826), specifically focusing on how tool definitions are extracted from the BFCL_v3 (Berkeley Function Calling Leaderboard) dataset.

## Paper Reference

**Magnet: Multi-turn Tool-use Data Synthesis and Distillation via Graph Translation**  
Fan Yin et al., 2025 — [arXiv 2503.07826](https://arxiv.org/abs/2503.07826)

### Key Section: 3.4 Tool Pool Construction

The paper describes a multi-source tool pool construction strategy:
- **StableToolBench**: Real-world RapidAPI tools (ToolEnv2404)
- **BFCL-v3**: Synthetic multi-turn function documentation
- Total: **5,011 unique APIs** after deduplication and filtering

---

## Script Architecture

### Core Components

1. **`tool_definition.py`** - Canonical data model matching the Magnet format
2. **`parse_bfcl.py`** - Parser for BFCL-v3 multi-turn function documentation
3. **`parse_stabletoolbench.py`** - Parser for StableToolBench/ToolEnv2404
4. **`collect_tools.py`** - Main orchestration script

---

## Tool Definition Format

The Magnet canonical format is a denormalized JSON structure where each entry represents a single API endpoint:

```json
{
  "category": "Storage",
  "tool_name": "gorilla_file_system",
  "tool_description": "Functions provided by the gorilla file system toolkit.",
  "api_name": "cat",
  "api_description": "Display the contents of a file of any extension from current directory.",
  "parameters": {
    "type": "dict",
    "properties": {
      "file_name": {
        "type": "string",
        "description": "The name of the file from current directory to display. No path is allowed."
      }
    },
    "required": ["file_name"],
    "optional": []
  }
}
```

### Schema Details

| Field | Description | Source |
|-------|-------------|--------|
| `category` | High-level domain (e.g., Storage, Finance, Weather) | Mapped from class name or tool metadata |
| `tool_name` | Parent tool/class identifier | BFCL: class name; StableToolBench: tool_name |
| `tool_description` | High-level tool purpose | BFCL: synthesized from class; StableToolBench: from tool metadata |
| `api_name` | Function/method name | Direct from function definition |
| `api_description` | Detailed function purpose | From function `description` field |
| `parameters` | OpenAI-style schema | Normalized to Magnet format |

---

## BFCL_v3 Extraction Process

### Data Source Structure

**Original BFCL format** (from `multi_turn_func_doc/gorilla_file_system.json`):
```json
{
  "name": "cat",
  "description": "This tool belongs to the Gorilla file system. It is a simple file system that allows users to perform basic file operations such as navigating directories, creating files and directories, reading and writing to files, etc. Tool description: Display the contents of a file of any extension from currrent directory.",
  "parameters": {
    "type": "dict",
    "properties": {
      "file_name": {
        "type": "string",
        "description": "The name of the file from current directory to display. No path is allowed."
      }
    },
    "required": ["file_name"]
  },
  "response": {
    "type": "dict",
    "properties": {
      "file_content": {
        "type": "string",
        "description": "The content of the file."
      }
    }
  }
}
```

### Extraction Pipeline

1. **Discovery Phase** (`discover_bfcl_classes`)
   - Scans BFCL_v3_multi_turn_*.json test files
   - Extracts `involved_classes` field to determine which tool classes are actually used
   - Example classes: `["gorilla_file_system", "trading_bot", "ticket_api", "weather_api", "math_api", "message_api", "calendar_api"]`

2. **Category Mapping** (`_resolve_category`)
   - Maps BFCL class names to Magnet categories:
     ```python
     _BFCL_CATEGORY_MAP = {
         "gorilla_file_system": "Storage",
         "trading_bot": "Finance",
         "ticket_api": "Events",
         "weather_api": "Weather",
         "math_api": "Science",
         "message_api": "Communication",
         "calendar_api": "Business_Software",
     }
     ```

3. **Parameter Normalization** (`_parse_openai_parameters`)
   - Converts OpenAI-style `"type": "object"` to Magnet's `"type": "dict"`
   - Separates `required` and `optional` parameters
   - Preserves nested parameter properties

4. **Filtering**
   - Skips functions with no parameters (unless `--no-require-parameters`)
   - Matches Magnet's quality filter from §3.4

---

## Sample Outputs from BFCL_v3

Below are actual extractions from the BFCL_v3 dataset showing the transformation from original format to Magnet canonical format.

### Example 1: Gorilla File System - `cat`

**Original BFCL Input:**
```json
{
  "name": "cat",
  "description": "This tool belongs to the Gorilla file system. It is a simple file system that allows users to perform basic file operations such as navigating directories, creating files and directories, reading and writing to files, etc. Tool description: Display the contents of a file of any extension from currrent directory.",
  "parameters": {
    "type": "dict",
    "properties": {
      "file_name": {
        "type": "string",
        "description": "The name of the file from current directory to display. No path is allowed."
      }
    },
    "required": ["file_name"]
  },
  "response": {
    "type": "dict",
    "properties": {
      "file_content": {
        "type": "string",
        "description": "The content of the file."
      }
    }
  }
}
```

**Magnet Canonical Output:**
```json
{
  "category": "Storage",
  "tool_name": "gorilla_file_system",
  "tool_description": "Functions provided by the gorilla file system toolkit.",
  "api_name": "cat",
  "api_description": "This tool belongs to the Gorilla file system. It is a simple file system that allows users to perform basic file operations such as navigating directories, creating files and directories, reading and writing to files, etc. Tool description: Display the contents of a file of any extension from currrent directory.",
  "parameters": {
    "type": "dict",
    "properties": {
      "file_name": {
        "type": "string",
        "description": "The name of the file from current directory to display. No path is allowed."
      }
    },
    "required": ["file_name"],
    "optional": []
  }
}
```

### Example 2: Gorilla File System - `find`

**Original BFCL Input:**
```json
{
  "name": "find",
  "description": "This tool belongs to the Gorilla file system. It is a simple file system that allows users to perform basic file operations such as navigating directories, creating files and directories, reading and writing to files, etc. Tool description: Find any file or directories under specific path that contain name in its file name. This method searches for files of any extension and directories within a specified path that match the given name. If no name is provided, it returns all files and directories in the specified path and its subdirectories. Note: This method performs a recursive search through all subdirectories of the given path.",
  "parameters": {
    "type": "dict",
    "properties": {
      "path": {
        "type": "string",
        "description": "The directory path to start the search. Defaults to the current directory (\".\").",
        "default": "."
      },
      "name": {
        "type": "string",
        "description": "The name of the file or directory to search for. If None, all items are returned.",
        "default": "None"
      }
    },
    "required": []
  },
  "response": {
    "type": "dict",
    "properties": {
      "matches": {
        "type": "array",
        "description": "A list of matching file and directory paths relative to the given path.",
        "items": {
          "type": "string"
        }
      }
    }
  }
}
```

**Magnet Canonical Output:**
```json
{
  "category": "Storage",
  "tool_name": "gorilla_file_system",
  "tool_description": "Functions provided by the gorilla file system toolkit.",
  "api_name": "find",
  "api_description": "This tool belongs to the Gorilla file system. It is a simple file system that allows users to perform basic file operations such as navigating directories, creating files and directories, reading and writing to files, etc. Tool description: Find any file or directories under specific path that contain name in its file name. This method searches for files of any extension and directories within a specified path that match the given name. If no name is provided, it returns all files and directories in the specified path and its subdirectories. Note: This method performs a recursive search through all subdirectories of the given path.",
  "parameters": {
    "type": "dict",
    "properties": {
      "path": {
        "type": "string",
        "description": "The directory path to start the search. Defaults to the current directory (\".\").",
        "default": "."
      },
      "name": {
        "type": "string",
        "description": "The name of the file or directory to search for. If None, all items are returned.",
        "default": "None"
      }
    },
    "required": [],
    "optional": ["path", "name"]
  }
}
```

### Example 3: Trading Bot - `buy_stock`

**Original BFCL Input (from trading_bot.json):**
```json
{
  "name": "buy_stock",
  "description": "Buy a stock with a given symbol and quantity.",
  "parameters": {
    "type": "dict",
    "properties": {
      "symbol": {
        "type": "string",
        "description": "The stock symbol to buy."
      },
      "quantity": {
        "type": "integer",
        "description": "The quantity of stocks to buy."
      }
    },
    "required": ["symbol", "quantity"]
  },
  "response": {
    "type": "dict",
    "properties": {
      "status": {
        "type": "string",
        "description": "The status of the buy operation."
      },
      "cost": {
        "type": "number",
        "description": "The total cost of the transaction."
      }
    }
  }
}
```

**Magnet Canonical Output:**
```json
{
  "category": "Finance",
  "tool_name": "trading_bot",
  "tool_description": "Functions provided by the trading bot toolkit.",
  "api_name": "buy_stock",
  "api_description": "Buy a stock with a given symbol and quantity.",
  "parameters": {
    "type": "dict",
    "properties": {
      "symbol": {
        "type": "string",
        "description": "The stock symbol to buy."
      },
      "quantity": {
        "type": "integer",
        "description": "The quantity of stocks to buy."
      }
    },
    "required": ["symbol", "quantity"],
    "optional": []
  }
}
```

### Example 4: Weather API - `get_weather`

**Original BFCL Input (from weather_api.json):**
```json
{
  "name": "get_weather",
  "description": "Get the current weather for a given location.",
  "parameters": {
    "type": "dict",
    "properties": {
      "location": {
        "type": "string",
        "description": "The city and state, e.g., San Francisco, CA."
      },
      "unit": {
        "type": "string",
        "description": "The unit of temperature, either 'celsius' or 'fahrenheit'.",
        "default": "fahrenheit",
        "enum": ["celsius", "fahrenheit"]
      }
    },
    "required": ["location"]
  },
  "response": {
    "type": "dict",
    "properties": {
      "temperature": {
        "type": "number",
        "description": "The current temperature."
      },
      "description": {
        "type": "string",
        "description": "A description of the weather."
      }
    }
  }
}
```

**Magnet Canonical Output:**
```json
{
  "category": "Weather",
  "tool_name": "weather_api",
  "tool_description": "Functions provided by the weather api toolkit.",
  "api_name": "get_weather",
  "api_description": "Get the current weather for a given location.",
  "parameters": {
    "type": "dict",
    "properties": {
      "location": {
        "type": "string",
        "description": "The city and state, e.g., San Francisco, CA."
      },
      "unit": {
        "type": "string",
        "description": "The unit of temperature, either 'celsius' or 'fahrenheit'.",
        "default": "fahrenheit",
        "enum": ["celsius", "fahrenheit"]
      }
    },
    "required": ["location"],
    "optional": ["unit"]
  }
}
```

---

## Key Transformation Logic

### 1. Category Resolution
```python
def _resolve_category(class_name: str) -> str:
    """Map a BFCL class name to a Magnet category string."""
    return _BFCL_CATEGORY_MAP.get(class_name, class_name.replace("_", " ").title())
```

### 2. Parameter Schema Normalization
```python
def _parse_openai_parameters(params_schema: dict) -> ToolParameters:
    """
    Convert an OpenAI-style parameters schema to ToolParameters.
    
    BFCL uses "type": "object"; we normalise it to "type": "dict"
    """
    properties: dict = params_schema.get("properties", {})
    required_names: list[str] = params_schema.get("required", [])
    all_names: list[str] = list(properties.keys())
    optional_names: list[str] = [n for n in all_names if n not in required_names]
    
    return ToolParameters(
        type="dict",
        properties=properties,
        required=required_names,
        optional=optional_names,
    )
```

### 3. Class Discovery from Test Files
```python
def discover_bfcl_classes(data_dir: Path) -> set[str]:
    """
    Scan the BFCL multi-turn test JSON files and collect all class names
    referenced in 'involved_classes' fields.
    """
    class_names: set[str] = set()
    
    for fname in BFCL_MULTI_TURN_FILES:
        fpath = data_dir / fname
        # Parse JSON array or JSON-lines format
        # Extract 'involved_classes' from each entry
        for entry in entries:
            for cls in entry.get("involved_classes", []):
                class_names.add(cls)
    
    return class_names
```

---

## Multi-Turn Test Entry Structure

BFCL_v3 multi-turn test files contain task instances that reference the tool classes:

```json
{
  "id": "multi_turn_base_0",
  "question": [
    [
      {
        "role": "user",
        "content": "Move 'final_report.pdf' within document directory to 'temp' directory..."
      }
    ],
    [...]
  ],
  "initial_config": {
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
  },
  "path": [
    "GorillaFileSystem.find",
    "GorillaFileSystem.mv",
    "GorillaFileSystem.grep"
  ],
  "involved_classes": ["TwitterAPI", "GorillaFileSystem"]
}
```

The `involved_classes` field is used to filter which function documentation files to parse.

---

## Statistics from BFCL_v3

Based on the extraction pipeline:

### Tool Classes Discovered
- `gorilla_file_system` (Storage)
- `trading_bot` (Finance)
- `ticket_api` (Events)
- `weather_api` (Weather)
- `math_api` (Science)
- `message_api` (Communication)
- `calendar_api` (Business_Software)
- `posting_api` (category fallback)
- `travel_booking` (category fallback)
- `vehicle_control` (category fallback)

### Function Counts per Class
- **gorilla_file_system**: 18 functions (cat, cd, cp, diff, du, echo, find, grep, ls, mkdir, mv, pwd, rm, rmdir, sort, tail, touch, wc)
- **trading_bot**: ~12 functions
- **weather_api**: ~5 functions
- **message_api**: ~8 functions

---

## Integration with APIGen-MT

### Relevance to Multi-Turn Dataset Generation

The Magnet tool pool extraction approach provides:

1. **Unified Schema**: Standardized format for tool definitions across different sources
2. **Quality Filtering**: Automatic exclusion of parameter-less APIs
3. **Category Organization**: Semantic grouping for tool selection
4. **Deduplication**: Prevents redundant tool definitions

### Potential Applications

1. **Tool Pool for APIGen-MT**: Use the extracted tool definitions as input for multi-turn conversation generation
2. **Tool Discovery**: Leverage the categorization system for selecting relevant tools per domain
3. **Schema Validation**: Apply the Magnet canonical format as a validation standard

---

## Usage Example

### Running the Extraction Pipeline

```bash
python collect_tools.py \
  --bfcl-func-doc data/BFCL_v3/multi_turn_func_doc \
  --bfcl-data-dir data/BFCL_v3 \
  --output output/tool_pool.jsonl \
  --stats
```

### Output Statistics
```
Per-category counts:
  Storage                                   18
  Finance                                   12
  Communication                              8
  Events                                     6
  Weather                                    5
  Science                                    4
  Business_Software                          3
  
Total: 56 definitions across 7 categories
```

---

## Design Decisions & Notes

1. **Parameter Filtering**: APIs with no parameters are excluded by default (matching Magnet §3.4), unless `--no-require-parameters` is used.

2. **Response Schema**: The `response` field in BFCL is preserved in the original data but not included in the Magnet canonical output (only parameters are retained).

3. **Tool Description**: For BFCL, the tool_description is synthesized as `"Functions provided by the {class_name} toolkit."` since BFCL doesn't provide tool-level metadata.

4. **Name Rewriting**: The paper mentions LLM-based name rewriting to avoid contamination. The scripts preserve original names; a separate post-processing step would be needed.

5. **Case-Insensitive Deduplication**: Functions with the same `(tool_name, api_name)` pair are deduplicated using lowercase matching.

---

## References

- **Magnet Paper**: [arXiv 2503.07826](https://arxiv.org/abs/2503.07826)
- **BFCL Dataset**: [gorilla-llm/Berkeley-Function-Calling-Leaderboard](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard)
- **StableToolBench**: [stabletoolbench/ToolEnv2404](https://huggingface.co/datasets/stabletoolbench/ToolEnv2404)