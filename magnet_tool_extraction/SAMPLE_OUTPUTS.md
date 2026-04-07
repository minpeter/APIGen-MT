# Sample Outputs from BFCL_v3 Extraction

This document shows actual extracted tool definitions from the BFCL_v3 dataset, demonstrating the transformation from BFCL format to Magnet canonical format.

## Extraction Summary

```
Dataset: BFCL_v3 (Berkeley Function Calling Leaderboard)
Source: gorilla-llm/Berkeley-Function-Calling-Leaderboard
Tools Extracted: 105
Categories: 8
```

---

## Category 1: Storage (17 tools)

### Sample Tool: cat

**Source**: `gorilla_file_system.json`

**Magnet Format**:
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

### Sample Tool: find (with optional parameters)

**Source**: `gorilla_file_system.json`

**Magnet Format**:
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

### All Storage Tools:
```
1. cat - Display file contents
2. cd - Change directory
3. cp - Copy files
4. diff - Compare files
5. du - Show disk usage
6. echo - Write to file
7. find - Find files/directories
8. ls - List directory contents
9. mkdir - Create directory
10. mv - Move/rename files
11. pwd - Print working directory
12. rm - Remove files
13. rmdir - Remove directory
14. sort - Sort file contents
15. tail - Display end of file
16. touch - Create empty file
17. write - Write content to file
```

---

## Category 2: Science (17 tools)

### Sample Tool: add

**Source**: `math_api.json`

**Magnet Format**:
```json
{
  "category": "Science",
  "tool_name": "math_api",
  "tool_description": "Functions provided by the math api toolkit.",
  "api_name": "add",
  "api_description": "This tool belongs to the Math API, which provides various mathematical operations. Tool description: Add two numbers.",
  "parameters": {
    "type": "dict",
    "properties": {
      "a": {
        "type": "float",
        "description": "First number."
      },
      "b": {
        "type": "float",
        "description": "Second number."
      }
    },
    "required": ["a", "b"],
    "optional": []
  }
}
```

### Sample Tool: log (with optional parameter)

**Source**: `math_api.json`

**Magnet Format**:
```json
{
  "category": "Science",
  "tool_name": "math_api",
  "tool_description": "Functions provided by the math api toolkit.",
  "api_name": "log",
  "api_description": "This tool belongs to the Math API, which provides various mathematical operations. Tool description: Calculate the logarithm of a number with a specified base.",
  "parameters": {
    "type": "dict",
    "properties": {
      "number": {
        "type": "float",
        "description": "The number to calculate the logarithm of."
      },
      "base": {
        "type": "float",
        "description": "The base of the logarithm. Defaults to base e (natural logarithm).",
        "default": 2.718281828459045
      }
    },
    "required": ["number"],
    "optional": ["base"]
  }
}
```

### All Science Tools:
```
1. absolute_value - Calculate absolute value
2. add - Add two numbers
3. cosine - Calculate cosine
4. divide - Divide two numbers
5. exponent - Calculate exponent
6. factorial - Calculate factorial
7. greatest_common_divisor - Calculate GCD
8. is_prime - Check if number is prime
9. lcm - Calculate least common multiple
10. log - Calculate logarithm
11. max - Find maximum
12. min - Find minimum
13. multiply - Multiply two numbers
14. power - Calculate power
15. remainder - Calculate remainder
16. sine - Calculate sine
17. square_root - Calculate square root
```

---

## Category 3: Finance (16 tools)

### Sample Tool: buy_stock

**Source**: `trading_bot.json`

**Magnet Format**:
```json
{
  "category": "Finance",
  "tool_name": "trading_bot",
  "tool_description": "Functions provided by the trading bot toolkit.",
  "api_name": "buy_stock",
  "api_description": "This tool belongs to the trading system, which allows users to trade stocks, manage their account, and view stock information. Tool description: Buy a stock with a given symbol and quantity.",
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

### All Finance Tools:
```
1. add_to_watchlist - Add stock to watchlist
2. cancel_order - Cancel an order
3. filter_stocks_by_price - Filter stocks by price
4. get_account_info - Get account information
5. get_stock_info - Get stock information
6. get_stock_price - Get stock price
7. get_watchlist - Get watchlist
8. limit_order - Place limit order
9. market_order - Place market order
10. remove_from_watchlist - Remove from watchlist
11. sell_stock - Sell stock
12. set_stop_loss - Set stop loss
13. set_take_profit - Set take profit
14. stock_performance - View stock performance
15. update_account - Update account
16. view_orders - View all orders
```

---

## Category 4: Vehicle Control (16 tools)

### Sample Tool: adjustClimateControl (with multiple optional parameters)

**Source**: `vehicle_control.json`

**Magnet Format**:
```json
{
  "category": "Vehicle Control",
  "tool_name": "vehicle_control_api",
  "tool_description": "Functions provided by the vehicle control api toolkit.",
  "api_name": "adjustClimateControl",
  "api_description": "This tool belongs to the vehicle control system, which allows users to control various aspects of a vehicle. Tool description: Adjust the climate control of the vehicle.",
  "parameters": {
    "type": "dict",
    "properties": {
      "temperature": {
        "type": "float",
        "description": "The desired temperature in Fahrenheit."
      },
      "unit": {
        "type": "string",
        "description": "The unit of temperature. Default is Fahrenheit.",
        "default": "Fahrenheit"
      },
      "mode": {
        "type": "string",
        "description": "The mode of the climate control system.",
        "default": "auto"
      },
      "fan_speed": {
        "type": "integer",
        "description": "The fan speed from 1 to 5.",
        "default": 3
      }
    },
    "required": ["temperature"],
    "optional": ["unit", "mode", "fan_speed"]
  }
}
```

### All Vehicle Control Tools:
```
1. activateParkingBrake - Activate parking brake
2. adjustClimateControl - Adjust climate
3. adjustMirror - Adjust mirrors
4. adjustSeat - Adjust seat position
5. controlSunroof - Control sunroof
6. controlWindows - Control windows
7. displayCarStatus - Display car status
8. estimate_distance - Estimate distance
9. fill_tank - Fill fuel tank
10. get_fuel_level - Get fuel level
11. get_speed - Get current speed
12. honk - Honk horn
13. set_speed - Set cruise control speed
14. start_engine - Start engine
15. steering_angle - Set steering angle
16. toggle_light - Toggle lights
```

---

## Category 5: Travel Booking (14 tools)

### Sample Tool: book_flight (complex parameters)

**Source**: `travel_booking.json`

**Magnet Format**:
```json
{
  "category": "Travel Booking",
  "tool_name": "travel_booking",
  "tool_description": "Functions provided by the travel booking toolkit.",
  "api_name": "book_flight",
  "api_description": "This tool belongs to the travel system, which allows users to book flights, manage their bookings, and view trip details. Tool description: Book a flight.",
  "parameters": {
    "type": "dict",
    "properties": {
      "origin": {
        "type": "string",
        "description": "The origin airport."
      },
      "destination": {
        "type": "string",
        "description": "The destination airport."
      },
      "date": {
        "type": "string",
        "description": "The date of the flight."
      },
      "airline": {
        "type": "string",
        "description": "The preferred airline."
      },
      "passengers": {
        "type": "integer",
        "description": "The number of passengers."
      },
      "class_type": {
        "type": "string",
        "description": "The class type (economy, business, first)."
      },
      "one_way": {
        "type": "boolean",
        "description": "Whether the flight is one-way."
      }
    },
    "required": ["origin", "destination", "date", "airline", "passengers", "class_type"],
    "optional": ["one_way"]
  }
}
```

---

## Category 6: Posting API (12 tools)

### Sample Tool: tweet

**Source**: `posting_api.json`

**Magnet Format**:
```json
{
  "category": "Posting Api",
  "tool_name": "posting_api",
  "tool_description": "Functions provided by the posting api toolkit.",
  "api_name": "tweet",
  "api_description": "This tool belongs to the TwitterAPI, which provides core functionality for posting tweets, retweeting, and managing user interactions on Twitter. Tool description: Post a tweet.",
  "parameters": {
    "type": "dict",
    "properties": {
      "text": {
        "type": "string",
        "description": "The text content of the tweet."
      },
      "media": {
        "type": "string",
        "description": "Optional media attachment for the tweet.",
        "default": "None"
      }
    },
    "required": ["text"],
    "optional": ["media"]
  }
}
```

---

## Category 7: Events (7 tools)

### Sample Tool: create_ticket

**Source**: `ticket_api.json`

**Magnet Format**:
```json
{
  "category": "Events",
  "tool_name": "ticket_api",
  "tool_description": "Functions provided by the ticket api toolkit.",
  "api_name": "create_ticket",
  "api_description": "This tool belongs to the ticketing system that is part of a company, which allows users to create, view, and manage support business tickets. Tool description: Create a ticket in the system and queue it.",
  "parameters": {
    "type": "dict",
    "properties": {
      "title": {
        "type": "string",
        "description": "Title of the ticket."
      },
      "description": {
        "type": "string",
        "description": "Description of the ticket. Defaults to an empty string.",
        "default": ""
      },
      "priority": {
        "type": "integer",
        "description": "Priority of the ticket, from 1 to 5. Defaults to 1. 5 is the highest priority.",
        "default": 1
      }
    },
    "required": ["title"],
    "optional": ["description", "priority"]
  }
}
```

---

## Category 8: Communication (6 tools)

### Sample Tool: delete_message

**Source**: `message_api.json`

**Magnet Format**:
```json
{
  "category": "Communication",
  "tool_name": "message_api",
  "tool_description": "Functions provided by the message api toolkit.",
  "api_name": "delete_message",
  "api_description": "This tool belongs to the Message API, which is used to manage user interactions in a workspace. Tool description: Delete the latest message sent to a receiver.",
  "parameters": {
    "type": "dict",
    "properties": {
      "receiver_id": {
        "type": "string",
        "description": "User ID of the user to send the message to."
      },
      "message_id": {
        "type": "integer",
        "description": "ID of the message to be deleted.",
        "default": -1
      }
    },
    "required": ["receiver_id"],
    "optional": ["message_id"]
  }
}
```

---

## Parameter Statistics

### Required Parameters
- **Total**: 147 required parameters across all tools
- **Average**: 1.4 required parameters per tool
- **Tools with 0 required**: 5 (5%)
- **Tools with 1 required**: 59 (56%)
- **Tools with 2+ required**: 41 (39%)

### Optional Parameters
- **Total**: 29 optional parameters across all tools
- **Average**: 0.28 optional parameters per tool
- **Tools with 0 optional**: 90 (86%)
- **Tools with 1+ optional**: 15 (14%)

### Parameter Types Distribution
```
string:    89 parameters (50%)
integer:   47 parameters (26%)
float:     32 parameters (18%)
boolean:   10 parameters (6%)
```

---

## Transformation Examples

### Example 1: Simple Transformation

**Input (BFCL JSONL)**:
```json
{"name": "ls", "description": "List directory contents", "parameters": {"type": "dict", "properties": {"path": {"type": "string", "description": "The path"}}, "required": []}}
```

**Output (Magnet Canonical)**:
```json
{
  "category": "Storage",
  "tool_name": "gorilla_file_system",
  "tool_description": "Functions provided by the gorilla file system toolkit.",
  "api_name": "ls",
  "api_description": "List directory contents",
  "parameters": {
    "type": "dict",
    "properties": {
      "path": {
        "type": "string",
        "description": "The path"
      }
    },
    "required": [],
    "optional": ["path"]
  }
}
```

### Example 2: Complex Transformation

**Input (BFCL JSONL)**:
```json
{"name": "book_flight", "description": "Book a flight", "parameters": {"type": "dict", "properties": {"origin": {"type": "string"}, "destination": {"type": "string"}, "date": {"type": "string"}, "airline": {"type": "string"}, "passengers": {"type": "integer"}}, "required": ["origin", "destination", "date", "airline", "passengers"]}}
```

**Output (Magnet Canonical)**:
```json
{
  "category": "Travel Booking",
  "tool_name": "travel_booking",
  "tool_description": "Functions provided by the travel booking toolkit.",
  "api_name": "book_flight",
  "api_description": "Book a flight",
  "parameters": {
    "type": "dict",
    "properties": {
      "origin": {"type": "string"},
      "destination": {"type": "string"},
      "date": {"type": "string"},
      "airline": {"type": "string"},
      "passengers": {"type": "integer"}
    },
    "required": ["origin", "destination", "date", "airline", "passengers"],
    "optional": []
  }
}
```

---

## Notes

1. All extracted tools include complete parameter schemas
2. Required and optional parameters are clearly separated
3. Default values are preserved for optional parameters
4. Category assignments follow the BFCL class structure
5. Tool descriptions are generated from class names when not provided