# BFCL Top-5 Tools by Category Analysis

Quick reference showing the top-5 most frequent tools for each category in the BFCL_v3 dataset.

## Quick Summary

- **1,674** unique tools in pool
- **3,401** total test invocations  
- **1,497** unique tools actually used in tests

## Categories with Pool Tools Only (No Test Usage Yet)

### 📂 Communication (10 tools)
**Available in Pool:**
1. `message_api.add_contact`
2. `message_api.delete_message`
3. `message_api.get_message_stats`
4. `message_api.get_user_id`
5. `message_api.list_users`

### 📂 Events (9 tools)
**Available in Pool:**
1. `ticket_api.close_ticket`
2. `ticket_api.create_ticket`
3. `ticket_api.edit_ticket`
4. `ticket_api.get_ticket`
5. `ticket_api.get_user_tickets`

### 📂 Finance (22 tools)
**Available in Pool:**
1. `trading_bot.add_to_watchlist`
2. `trading_bot.cancel_order`
3. `trading_bot.filter_stocks_by_price`
4. `trading_bot.fund_account`
5. `trading_bot.get_account_info`

### 📂 Posting API (14 tools)
**Available in Pool:**
1. `posting_api.authenticate_twitter`
2. `posting_api.comment`
3. `posting_api.follow_user`
4. `posting_api.get_tweet`
5. `posting_api.get_tweet_comments`

### 📂 Science (17 tools)
**Available in Pool:**
1. `math_api.absolute_value`
2. `math_api.add`
3. `math_api.divide`
4. `math_api.imperial_si_conversion`
5. `math_api.logarithm`

### 📂 Storage (18 tools)
**Available in Pool:**
1. `gorilla_file_system.cat`
2. `gorilla_file_system.cd`
3. `gorilla_file_system.cp`
4. `gorilla_file_system.diff`
5. `gorilla_file_system.du`

### 📂 Travel Booking (17 tools)
**Available in Pool:**
1. `travel_booking.authenticate_travel`
2. `travel_booking.book_flight`
3. `travel_booking.cancel_booking`
4. `travel_booking.compute_exchange_rate`
5. `travel_booking.contact_customer_support`

### 📂 Vehicle Control (22 tools)
**Available in Pool:**
1. `vehicle_control.activateParkingBrake`
2. `vehicle_control.adjustClimateControl`
3. `vehicle_control.check_tire_pressure`
4. `vehicle_control.displayCarStatus`
5. `vehicle_control.display_log`

## Categories with Active Test Usage

### 📂 Multiple Functions (1,915 invocations, 701 unique tools)

**Top-5 Most Used:**

| Rank | Tool Name | Count | Percentage |
|------|-----------|-------|------------|
| 1 | `Events_3_FindEvents` | 89 | 4.6% |
| 2 | `Movies_3_FindMovies` | 76 | 4.0% |
| 3 | `Music_3_LookupMusic` | 62 | 3.2% |
| 4 | `Weather_1_GetWeather` | 40 | 2.1% |
| 5 | `Movies_1_FindMovies` | 37 | 1.9% |

### 📂 Simple Functions (658 invocations, 453 unique tools)

**Top-5 Most Used:**

| Rank | Tool Name | Count | Percentage |
|------|-----------|-------|------------|
| 1 | `cmd_controller.execute` | 28 | 4.3% |
| 2 | `get_current_weather` | 20 | 3.0% |
| 3 | `Movies_3_FindMovies` | 18 | 2.7% |
| 4 | `Weather_1_GetWeather` | 16 | 2.4% |
| 5 | `requests.get` | 11 | 1.7% |

### 📂 Parallel Functions (577 invocations, 192 unique tools)

**Top-5 Most Used:**

| Rank | Tool Name | Count | Percentage |
|------|-----------|-------|------------|
| 1 | `get_current_weather` | 23 | 4.0% |
| 2 | `log_food` | 10 | 1.7% |
| 3 | `calculate_bmi` | 10 | 1.7% |
| 4 | `math.factorial` | 10 | 1.7% |
| 5 | `array_sort` | 8 | 1.4% |

### 📂 Java (150 invocations, 150 unique tools)

All tools used exactly once (0.7% each). Examples:
1. `GeometryPresentation.createPresentation`
2. `SQLCompletionAnalyzer.makeProposalsFromObject`
3. `FireBirdUtils.getViewSourceWithHeader`

### 📂 SQL (101 invocations, 1 unique tool)

**Only Tool Used:**
| Rank | Tool Name | Count | Percentage |
|------|-----------|-------|------------|
| 1 | `sql.execute` | 101 | 100.0% |

## Key Insights

### Popular Across Multiple Categories
These tools appear in multiple test categories:
- **`get_current_weather`** - Simple Functions (#2), Parallel Functions (#1)
- **`Weather_1_GetWeather`** - Multiple Functions (#4), Simple Functions (#4)
- **`Movies_3_FindMovies`** - Multiple Functions (#2), Simple Functions (#3)

### Category Characteristics
- **SQL**: Single tool (`sql.execute`) used in all 101 test cases
- **Java**: Every test uses a different tool (150 unique tools for 150 invocations)
- **Multiple Functions**: Highest diversity (701 unique tools for 1,915 invocations)

### Unused Categories
Several categories have tool definitions but **zero test usage**:
- Communication, Events, Finance, Posting API, Science, Storage, Travel Booking, Vehicle Control

These may be newly added categories awaiting test case development.

## Generated Files

- **`TOP5_TOOLS_BY_CATEGORY.md`** - This file
- **`bfcl_tool_frequency_analysis.json`** - Full analysis data in JSON format

## Analysis Scripts

- `analyze_bfcl_tool_frequency.py` - Main analysis script
- `generate_top5_table.py` - Table generation script

---

**Generated:** 2025-01-15  
**Data Source:** BFCL_v3 dataset  
**Analysis Version:** 1.0