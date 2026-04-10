# BFCL Tool Pool: Top-5 Tools by Category Analysis

## Overview

This analysis extracts and displays the top-5 most frequent tools for each category in the BFCL (Berkeley Function Calling Leaderboard) tool pool.

**Data Source**: `/home/ishalyminov/data/APIGen-MT/magnet_tool_extraction/bfcl_v3_all_tool_definitions.jsonl`

## Summary Statistics

- **Total Tools**: 1,674
- **Total Categories**: 9
- **Analysis Type**: Tool definitions (not invocations)

Note: Each tool appears exactly once in the pool (these are tool definitions, not usage counts).

## Top-5 Tools by Category

### 1. Communication (10 tools)

**Top-5:**
1. `message_api.add_contact`
2. `message_api.delete_message`
3. `message_api.get_message_stats`
4. `message_api.get_user_id`
5. `message_api.list_users`

---

### 2. Events (9 tools)

**Top-5:**
1. `ticket_api.close_ticket`
2. `ticket_api.create_ticket`
3. `ticket_api.edit_ticket`
4. `ticket_api.get_ticket`
5. `ticket_api.get_user_tickets`

---

### 3. Finance (22 tools)

**Top-5:**
1. `trading_bot.add_to_watchlist`
2. `trading_bot.cancel_order`
3. `trading_bot.filter_stocks_by_price`
4. `trading_bot.fund_account`
5. `trading_bot.get_account_info`

---

### 4. Posting API (14 tools)

**Top-5:**
1. `posting_api.authenticate_twitter`
2. `posting_api.comment`
3. `posting_api.follow_user`
4. `posting_api.get_tweet`
5. `posting_api.get_tweet_comments`

---

### 5. Science (17 tools)

**Top-5:**
1. `math_api.absolute_value`
2. `math_api.add`
3. `math_api.divide`
4. `math_api.imperial_si_conversion`
5. `math_api.logarithm`

---

### 6. Storage (18 tools)

**Top-5:**
1. `gorilla_file_system.cat`
2. `gorilla_file_system.cd`
3. `gorilla_file_system.cp`
4. `gorilla_file_system.diff`
5. `gorilla_file_system.du`

---

### 7. Travel Booking (17 tools)

**Top-5:**
1. `travel_booking.authenticate_travel`
2. `travel_booking.book_flight`
3. `travel_booking.cancel_booking`
4. `travel_booking.compute_exchange_rate`
5. `travel_booking.contact_customer_support`

---

### 8. Vehicle Control (22 tools)

**Top-5:**
1. `vehicle_control.activateParkingBrake`
2. `vehicle_control.adjustClimateControl`
3. `vehicle_control.check_tire_pressure`
4. `vehicle_control.displayCarStatus`
5. `vehicle_control.display_log`

---

### 9. Unknown (1,545 tools)

This category contains general-purpose and domain-specific tools.

**Top-5:**
1. `calculate_triangle_area`
2. `math.factorial`
3. `math.hypot`
4. `algebra.quadratic_roots`
5. `solve_quadratic_equation`

---

## Category Distribution

```
Communication     :   10 tools  ▏
Events           :    9 tools  ▏
Finance          :   22 tools  █
Posting Api      :   14 tools  ▎
Science          :   17 tools  ▊
Storage          :   18 tools  ▊
Travel Booking   :   17 tools  ▊
Vehicle Control  :   22 tools  █
Unknown          : 1545 tools  ████████████████████████████████████████████████
```

## Tool Naming Patterns

### Structured APIs (with tool_name prefix)
Most categories use a consistent naming pattern: `{tool_name}.{api_name}`

- **Communication**: `message_api.*`
- **Events**: `ticket_api.*`
- **Finance**: `trading_bot.*`
- **Posting API**: `posting_api.*`
- **Science**: `math_api.*`
- **Storage**: `gorilla_file_system.*`
- **Travel Booking**: `travel_booking.*`
- **Vehicle Control**: `vehicle_control.*`

### Unstructured Tools (Unknown category)
The "Unknown" category contains diverse tools with varying naming conventions:
- Standalone functions: `calculate_triangle_area`, `solve_quadratic_equation`
- Module functions: `math.factorial`, `math.hypot`, `algebra.quadratic_roots`
- Domain-specific tools

## Analysis Scripts

### Main Script
- **`analyze_tool_pool_top5.py`** - Extracts top-5 tools per category from the tool pool

### Output Files
- **`TOOL_POOL_TOP5_BY_CATEGORY.md`** - Markdown report
- **`tool_pool_top5_analysis.json`** - Machine-readable JSON results

### Usage
```bash
cd /home/ishalyminov/data/APIGen-MT/magnet_tool_extraction
python3 analyze_tool_pool_top5.py
```

## Key Observations

1. **Dominant Category**: The "Unknown" category contains 92% of all tools (1,545 out of 1,674)

2. **Structured Categories**: 8 well-defined categories with consistent naming patterns:
   - Average 16 tools per structured category
   - All use `{tool_name}.{api_name}` format

3. **Tool Definition vs. Usage**: This analysis shows tool availability, not usage frequency
   - Each tool appears exactly once in the pool
   - For actual usage statistics, test case analysis would be needed

4. **Category Completeness**: All listed categories have tools defined in the pool

## Related Files

- **Tool Pool**: `bfcl_v3_all_tool_definitions.jsonl`
- **Test Data**: `/home/ishalyminov/data/magnet_mt/data/BFCL_v3/`
- **Previous Analysis**: See `analyze_bfcl_tool_frequency.py` for usage analysis

---

**Generated**: 2025-01-15  
**Script**: `analyze_tool_pool_top5.py`  
**Data Source**: BFCL_v3 tool pool