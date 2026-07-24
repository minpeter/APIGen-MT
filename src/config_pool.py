"""
Domain-specific initial config pools for diverse trajectory generation.

Each domain has a list of config variations representing realistic scenarios.
generate_random_config() picks one variation per domain and combines them
into a full config dict suitable for create_python_tool_instances().

To revert to original behavior, set USE_CONFIG_POOL = False in tool_manager.py.

CHANGELOG:
- 2026-06-11: Created config_pool.py with domain-specific config pools
  - VehicleControlAPI: 10 varied scenarios (parked, driving, cold morning, hill, etc.)
  - TradingBot: 5 varied accounts (admin, retail, fund manager, penny stock, day trader)
  - TicketAPI: 5 varied queues (support, admin, HR, ops, empty new agent)
  - TravelAPI: 5 varied profiles (standard, business, luxury, budget, corporate)
  - PostingAPI: 5 varied personas (tech, foodie, travel, newcomer, sports)
  - MessageAPI: 5 varied workspaces (standard, dev team, classroom, family, startup)
  - GorillaFileSystem: 5 varied filesystems (standard, code project, photos, server, research)
- Modified tool_manager.py:
  - Renamed FULL_INITIAL_CONFIGS -> FULL_INITIAL_CONFIGS_ORIGINAL (kept as backup)
  - Added USE_CONFIG_POOL = True flag at top of file
  - reset_python_tool_instances() now calls generate_random_config() when pool enabled
  - Backup of original file at tool_manager.py.bak
"""

import copy
import random
from typing import Dict, List, Any


_LOCKED_ALL = {"driver": "locked", "passenger": "locked", "rear_left": "locked", "rear_right": "locked"}
_UNLOCKED_ALL = {"driver": "unlocked", "passenger": "unlocked", "rear_left": "unlocked", "rear_right": "unlocked"}

VEHICLE_CONFIGS: List[Dict[str, Any]] = [
    {
        "remainingUnlockedDoors": 0, "fuelLevel": 15.0, "batteryVoltage": 12.8,
        "engineState": "stopped", "doorStatus": copy.deepcopy(_LOCKED_ALL),
        "acTemperature": 22.0, "fanSpeed": 60, "acMode": "auto",
        "humidityLevel": 45.0, "headLightStatus": "off",
        "parkingBrakeStatus": "released", "parkingBrakeForce": 0.0,
        "slopeAngle": 0.0, "distanceToNextVehicle": 100.0,
        "cruiseStatus": "inactive", "destination": "Grand Canyon",
        "frontLeftTirePressure": 35.0, "frontRightTirePressure": 35.0,
        "rearLeftTirePressure": 35.0, "rearRightTirePressure": 35.0,
    },
    {
        "remainingUnlockedDoors": 0, "fuelLevel": 55.0, "batteryVoltage": 12.6,
        "engineState": "stopped", "doorStatus": copy.deepcopy(_LOCKED_ALL),
        "acTemperature": 28.0, "fanSpeed": 40, "acMode": "cool",
        "humidityLevel": 60.0, "headLightStatus": "off",
        "parkingBrakeStatus": "engaged", "parkingBrakeForce": 50.0,
        "slopeAngle": 0.0, "distanceToNextVehicle": 100.0,
        "cruiseStatus": "inactive", "destination": "None",
        "frontLeftTirePressure": 33.0, "frontRightTirePressure": 33.0,
        "rearLeftTirePressure": 34.0, "rearRightTirePressure": 34.0,
    },
    {
        "remainingUnlockedDoors": 4, "fuelLevel": 30.0, "batteryVoltage": 12.4,
        "engineState": "running", "doorStatus": copy.deepcopy(_UNLOCKED_ALL),
        "acTemperature": 24.0, "fanSpeed": 50, "acMode": "auto",
        "humidityLevel": 50.0, "headLightStatus": "off",
        "parkingBrakeStatus": "released", "parkingBrakeForce": 0.0,
        "slopeAngle": 0.0, "distanceToNextVehicle": 100.0,
        "cruiseStatus": "inactive", "destination": "None",
        "frontLeftTirePressure": 35.0, "frontRightTirePressure": 35.0,
        "rearLeftTirePressure": 35.0, "rearRightTirePressure": 35.0,
    },
    {
        "remainingUnlockedDoors": 0, "fuelLevel": 5.0, "batteryVoltage": 11.9,
        "engineState": "stopped", "doorStatus": copy.deepcopy(_LOCKED_ALL),
        "acTemperature": 22.0, "fanSpeed": 30, "acMode": "auto",
        "humidityLevel": 40.0, "headLightStatus": "off",
        "parkingBrakeStatus": "released", "parkingBrakeForce": 0.0,
        "slopeAngle": 0.0, "distanceToNextVehicle": 100.0,
        "cruiseStatus": "inactive", "destination": "None",
        "frontLeftTirePressure": 32.0, "frontRightTirePressure": 31.0,
        "rearLeftTirePressure": 33.0, "rearRightTirePressure": 32.0,
    },
    {
        "remainingUnlockedDoors": 0, "fuelLevel": 35.0, "batteryVoltage": 13.0,
        "engineState": "running", "doorStatus": copy.deepcopy(_LOCKED_ALL),
        "acTemperature": 20.0, "fanSpeed": 70, "acMode": "auto",
        "humidityLevel": 55.0, "headLightStatus": "on",
        "parkingBrakeStatus": "released", "parkingBrakeForce": 0.0,
        "slopeAngle": 0.0, "distanceToNextVehicle": 45.0,
        "cruiseStatus": "inactive", "destination": "Chicago",
        "frontLeftTirePressure": 36.0, "frontRightTirePressure": 36.0,
        "rearLeftTirePressure": 36.0, "rearRightTirePressure": 36.0,
    },
    {
        "remainingUnlockedDoors": 0, "fuelLevel": 40.0, "batteryVoltage": 12.1,
        "engineState": "stopped", "doorStatus": copy.deepcopy(_LOCKED_ALL),
        "acTemperature": 2.0, "fanSpeed": 80, "acMode": "heat",
        "humidityLevel": 85.0, "headLightStatus": "off",
        "parkingBrakeStatus": "engaged", "parkingBrakeForce": 55.0,
        "slopeAngle": 2.0, "distanceToNextVehicle": 100.0,
        "cruiseStatus": "inactive", "destination": "None",
        "frontLeftTirePressure": 30.0, "frontRightTirePressure": 30.0,
        "rearLeftTirePressure": 30.0, "rearRightTirePressure": 31.0,
    },
    {
        "remainingUnlockedDoors": 4, "fuelLevel": 25.0, "batteryVoltage": 12.7,
        "engineState": "running", "doorStatus": copy.deepcopy(_UNLOCKED_ALL),
        "acTemperature": 23.0, "fanSpeed": 45, "acMode": "auto",
        "humidityLevel": 48.0, "headLightStatus": "off",
        "parkingBrakeStatus": "released", "parkingBrakeForce": 0.0,
        "slopeAngle": 0.0, "distanceToNextVehicle": 100.0,
        "cruiseStatus": "inactive", "destination": "None",
        "frontLeftTirePressure": 34.0, "frontRightTirePressure": 34.0,
        "rearLeftTirePressure": 35.0, "rearRightTirePressure": 35.0,
    },
    {
        "remainingUnlockedDoors": 0, "fuelLevel": 45.0, "batteryVoltage": 12.9,
        "engineState": "running", "doorStatus": copy.deepcopy(_LOCKED_ALL),
        "acTemperature": 22.0, "fanSpeed": 55, "acMode": "auto",
        "humidityLevel": 42.0, "headLightStatus": "auto",
        "parkingBrakeStatus": "released", "parkingBrakeForce": 0.0,
        "slopeAngle": 0.0, "distanceToNextVehicle": 100.0,
        "cruiseStatus": "inactive", "destination": "None",
        "frontLeftTirePressure": 35.0, "frontRightTirePressure": 35.0,
        "rearLeftTirePressure": 35.0, "rearRightTirePressure": 35.0,
    },
    {
        "remainingUnlockedDoors": 0, "fuelLevel": 20.0, "batteryVoltage": 12.5,
        "engineState": "running", "doorStatus": copy.deepcopy(_LOCKED_ALL),
        "acTemperature": 21.0, "fanSpeed": 65, "acMode": "cool",
        "humidityLevel": 52.0, "headLightStatus": "auto",
        "parkingBrakeStatus": "released", "parkingBrakeForce": 0.0,
        "slopeAngle": 0.0, "distanceToNextVehicle": 25.0,
        "cruiseStatus": "active", "destination": "Denver",
        "frontLeftTirePressure": 36.0, "frontRightTirePressure": 35.0,
        "rearLeftTirePressure": 36.0, "rearRightTirePressure": 35.0,
    },
    {
        "remainingUnlockedDoors": 0, "fuelLevel": 50.0, "batteryVoltage": 12.6,
        "engineState": "stopped", "doorStatus": copy.deepcopy(_LOCKED_ALL),
        "acTemperature": 22.0, "fanSpeed": 50, "acMode": "auto",
        "humidityLevel": 44.0, "headLightStatus": "off",
        "parkingBrakeStatus": "released", "parkingBrakeForce": 0.0,
        "slopeAngle": 12.0, "distanceToNextVehicle": 100.0,
        "cruiseStatus": "inactive", "destination": "None",
        "frontLeftTirePressure": 35.0, "frontRightTirePressure": 35.0,
        "rearLeftTirePressure": 35.0, "rearRightTirePressure": 35.0,
    },
]


TRADING_CONFIGS: List[Dict[str, Any]] = [
    {
        "account_info": {"account_id": 12345, "balance": 10000.0, "binding_card": 1974202140965533},
        "authenticated": False, "market_status": "Open", "order_counter": 12446,
        "username": "trader_admin", "password": "testpass",
        "stocks": {
            "AAPL": {"price": 227.16, "percent_change": 0.17, "volume": 2.552, "MA(5)": 227.11, "MA(20)": 227.09},
            "GOOG": {"price": 2840.34, "percent_change": 0.24, "volume": 1.123, "MA(5)": 2835.67, "MA(20)": 2842.15},
            "TSLA": {"price": 667.92, "percent_change": -0.12, "volume": 1.654, "MA(5)": 671.15, "MA(20)": 668.2},
            "MSFT": {"price": 310.23, "percent_change": 0.09, "volume": 3.234, "MA(5)": 309.88, "MA(20)": 310.11},
            "NVDA": {"price": 220.34, "percent_change": 0.34, "volume": 1.234, "MA(5)": 220.45, "MA(20)": 220.67},
            "ALPH": {"price": 1320.45, "percent_change": -0.08, "volume": 1.567, "MA(5)": 1321.12, "MA(20)": 1325.78},
            "OMEG": {"price": 457.23, "percent_change": 0.12, "volume": 2.345, "MA(5)": 456.78, "MA(20)": 458.12},
            "QUAS": {"price": 725.89, "percent_change": -0.03, "volume": 1.789, "MA(5)": 726.45, "MA(20)": 728.0},
            "NEPT": {"price": 88.34, "percent_change": 0.19, "volume": 0.654, "MA(5)": 88.21, "MA(20)": 88.67},
            "SYNX": {"price": 345.67, "percent_change": 0.11, "volume": 2.112, "MA(5)": 345.34, "MA(20)": 346.12},
            "ZETA": {"price": 150.45, "percent_change": 0.05, "volume": 1.789, "MA(5)": 150.0, "MA(20)": 149.5},
        },
        "watch_list": ["NVDA", "ZETA"],
        "transaction_history": [{"order_id": 12346, "symbol": "GOOG", "price": 2840.34, "num_shares": 5, "status": "Pending", "timestamp": "2024-10-27 14:10:53"}],
    },
    {
        "account_info": {"account_id": 67890, "balance": 2500.0, "binding_card": 4532123498765432},
        "authenticated": False, "market_status": "Closed", "order_counter": 500,
        "username": "retail_trader", "password": "testpass",
        "stocks": {
            "AAPL": {"price": 229.50, "percent_change": -0.32, "volume": 3.100, "MA(5)": 230.10, "MA(20)": 228.90},
            "GOOG": {"price": 2855.00, "percent_change": 0.51, "volume": 0.980, "MA(5)": 2850.20, "MA(20)": 2848.00},
            "TSLA": {"price": 645.10, "percent_change": -3.41, "volume": 4.200, "MA(5)": 660.50, "MA(20)": 672.00},
            "MSFT": {"price": 315.80, "percent_change": 1.84, "volume": 2.800, "MA(5)": 312.40, "MA(20)": 311.00},
            "NVDA": {"price": 245.60, "percent_change": 11.50, "volume": 5.600, "MA(5)": 230.00, "MA(20)": 225.00},
            "ALPH": {"price": 1290.00, "percent_change": -2.30, "volume": 0.890, "MA(5)": 1310.00, "MA(20)": 1335.00},
            "OMEG": {"price": 470.50, "percent_change": 2.90, "volume": 1.700, "MA(5)": 462.00, "MA(20)": 458.00},
            "QUAS": {"price": 740.20, "percent_change": 1.97, "volume": 1.100, "MA(5)": 730.00, "MA(20)": 725.00},
            "NEPT": {"price": 92.10, "percent_change": 4.26, "volume": 0.450, "MA(5)": 89.00, "MA(20)": 87.00},
            "SYNX": {"price": 358.90, "percent_change": 3.80, "volume": 1.900, "MA(5)": 348.00, "MA(20)": 342.00},
            "ZETA": {"price": 162.30, "percent_change": 7.86, "volume": 2.400, "MA(5)": 155.00, "MA(20)": 148.00},
        },
        "watch_list": ["AAPL", "TSLA", "NVDA"],
        "transaction_history": [],
    },
    {
        "account_info": {"account_id": 11111, "balance": 50000.0, "binding_card": 5555666677778888},
        "authenticated": False, "market_status": "Open", "order_counter": 89000,
        "username": "fund_manager", "password": "testpass",
        "stocks": {
            "AAPL": {"price": 225.00, "percent_change": -1.08, "volume": 4.500, "MA(5)": 227.50, "MA(20)": 230.00},
            "GOOG": {"price": 2820.00, "percent_change": -0.71, "volume": 1.500, "MA(5)": 2835.00, "MA(20)": 2850.00},
            "TSLA": {"price": 680.00, "percent_change": 1.81, "volume": 2.200, "MA(5)": 670.00, "MA(20)": 665.00},
            "MSFT": {"price": 308.00, "percent_change": -0.72, "volume": 3.800, "MA(5)": 310.00, "MA(20)": 312.00},
            "NVDA": {"price": 215.00, "percent_change": -2.43, "volume": 2.000, "MA(5)": 220.00, "MA(20)": 225.00},
            "ALPH": {"price": 1350.00, "percent_change": 2.24, "volume": 0.900, "MA(5)": 1325.00, "MA(20)": 1310.00},
            "OMEG": {"price": 445.00, "percent_change": -2.69, "volume": 1.200, "MA(5)": 455.00, "MA(20)": 460.00},
            "QUAS": {"price": 710.00, "percent_change": -2.19, "volume": 0.800, "MA(5)": 725.00, "MA(20)": 735.00},
            "NEPT": {"price": 85.00, "percent_change": -3.77, "volume": 0.350, "MA(5)": 88.00, "MA(20)": 90.00},
            "SYNX": {"price": 340.00, "percent_change": -1.64, "volume": 1.500, "MA(5)": 345.00, "MA(20)": 350.00},
            "ZETA": {"price": 148.00, "percent_change": -1.62, "volume": 1.200, "MA(5)": 150.00, "MA(20)": 152.00},
        },
        "watch_list": ["ALPH", "OMEG", "QUAS"],
        "transaction_history": [
            {"order_id": 89001, "symbol": "AAPL", "price": 225.00, "num_shares": 100, "status": "Filled", "timestamp": "2024-11-01 09:35:00"},
            {"order_id": 89002, "symbol": "TSLA", "price": 680.00, "num_shares": 50, "status": "Filled", "timestamp": "2024-11-01 10:15:00"},
            {"order_id": 89003, "symbol": "NVDA", "price": 215.00, "num_shares": 200, "status": "Pending", "timestamp": "2024-11-01 14:00:00"},
        ],
    },
    {
        "account_info": {"account_id": 22222, "balance": 150.0, "binding_card": 1111222233334444},
        "authenticated": False, "market_status": "Open", "order_counter": 7,
        "username": "penny_stock_fan", "password": "testpass",
        "stocks": {
            "AAPL": {"price": 228.00, "percent_change": 0.44, "volume": 2.800, "MA(5)": 227.00, "MA(20)": 226.00},
            "GOOG": {"price": 2845.00, "percent_change": 0.17, "volume": 1.050, "MA(5)": 2842.00, "MA(20)": 2838.00},
            "TSLA": {"price": 660.00, "percent_change": -1.19, "volume": 1.900, "MA(5)": 668.00, "MA(20)": 672.00},
            "MSFT": {"price": 312.00, "percent_change": 0.57, "volume": 3.100, "MA(5)": 310.50, "MA(20)": 309.00},
            "NVDA": {"price": 230.00, "percent_change": 4.39, "volume": 2.500, "MA(5)": 222.00, "MA(20)": 218.00},
            "ALPH": {"price": 1330.00, "percent_change": 0.72, "volume": 1.200, "MA(5)": 1325.00, "MA(20)": 1320.00},
            "OMEG": {"price": 460.00, "percent_change": 0.61, "volume": 1.800, "MA(5)": 458.00, "MA(20)": 455.00},
            "QUAS": {"price": 730.00, "percent_change": 0.57, "volume": 1.300, "MA(5)": 727.00, "MA(20)": 724.00},
            "NEPT": {"price": 90.00, "percent_change": 1.88, "volume": 0.800, "MA(5)": 88.50, "MA(20)": 87.00},
            "SYNX": {"price": 350.00, "percent_change": 1.25, "volume": 1.600, "MA(5)": 347.00, "MA(20)": 344.00},
            "ZETA": {"price": 155.00, "percent_change": 3.02, "volume": 1.500, "MA(5)": 151.00, "MA(20)": 148.00},
        },
        "watch_list": ["NEPT", "ZETA"],
        "transaction_history": [
            {"order_id": 5, "symbol": "NEPT", "price": 88.00, "num_shares": 1, "status": "Filled", "timestamp": "2024-10-20 10:00:00"},
        ],
    },
    {
        "account_info": {"account_id": 33333, "balance": 75000.0, "binding_card": 9876543210987654},
        "authenticated": False, "market_status": "Open", "order_counter": 15000,
        "username": "day_trader_pro", "password": "testpass",
        "stocks": {
            "AAPL": {"price": 230.50, "percent_change": 1.47, "volume": 5.200, "MA(5)": 228.00, "MA(20)": 225.00},
            "GOOG": {"price": 2870.00, "percent_change": 1.05, "volume": 1.800, "MA(5)": 2850.00, "MA(20)": 2835.00},
            "TSLA": {"price": 690.00, "percent_change": 3.31, "volume": 3.500, "MA(5)": 670.00, "MA(20)": 660.00},
            "MSFT": {"price": 320.00, "percent_change": 3.14, "volume": 4.100, "MA(5)": 312.00, "MA(20)": 308.00},
            "NVDA": {"price": 250.00, "percent_change": 13.44, "volume": 6.200, "MA(5)": 228.00, "MA(20)": 218.00},
            "ALPH": {"price": 1280.00, "percent_change": -3.05, "volume": 1.100, "MA(5)": 1320.00, "MA(20)": 1340.00},
            "OMEG": {"price": 480.00, "percent_change": 4.99, "volume": 2.100, "MA(5)": 460.00, "MA(20)": 450.00},
            "QUAS": {"price": 750.00, "percent_change": 3.33, "volume": 1.400, "MA(5)": 730.00, "MA(20)": 720.00},
            "NEPT": {"price": 95.00, "percent_change": 7.55, "volume": 1.100, "MA(5)": 88.00, "MA(20)": 84.00},
            "SYNX": {"price": 365.00, "percent_change": 5.60, "volume": 2.400, "MA(5)": 350.00, "MA(20)": 340.00},
            "ZETA": {"price": 165.00, "percent_change": 9.67, "volume": 2.800, "MA(5)": 152.00, "MA(20)": 145.00},
        },
        "watch_list": ["AAPL", "NVDA", "TSLA", "NEPT", "ZETA"],
        "transaction_history": [
            {"order_id": 15001, "symbol": "NVDA", "price": 240.00, "num_shares": 50, "status": "Filled", "timestamp": "2024-11-05 09:32:00"},
            {"order_id": 15002, "symbol": "AAPL", "price": 229.00, "num_shares": 30, "status": "Filled", "timestamp": "2024-11-05 09:45:00"},
            {"order_id": 15003, "symbol": "TSLA", "price": 688.00, "num_shares": 20, "status": "Pending", "timestamp": "2024-11-05 10:00:00"},
        ],
    },
]


TICKET_CONFIGS: List[Dict[str, Any]] = [
    {
        "tickets_queue": [
            {"id": 123456, "title": "System Error", "description": "There is a critical system error that needs immediate attention.", "status": "Open", "priority": 4, "created_by": "support_agent"},
            {"id": 654321, "title": "Feature Request", "description": "Request for a new feature in the application.", "status": "In Progress", "priority": 2, "created_by": "support_agent"},
        ],
        "ticket_count": 654321, "ticket_counter": 654321,
        "current_user": "", "authenticated": False,
        "username": "support_agent", "password": "testpass",
    },
    {
        "tickets_queue": [
            {"id": 1001, "title": "Login failure", "description": "Users unable to log in after password reset.", "status": "Open", "priority": 4, "created_by": "admin_user"},
            {"id": 1002, "title": "Database timeout", "description": "Intermittent timeouts on production database.", "status": "Open", "priority": 5, "created_by": "admin_user"},
            {"id": 1003, "title": "Email notifications delayed", "description": "Outbound emails are queued but not sending.", "status": "In Progress", "priority": 3, "created_by": "admin_user"},
            {"id": 1004, "title": "UI glitch on dashboard", "description": "Charts not rendering correctly on Safari.", "status": "Open", "priority": 2, "created_by": "admin_user"},
        ],
        "ticket_count": 1004, "ticket_counter": 1004,
        "current_user": "", "authenticated": False,
        "username": "admin_user", "password": "testpass",
    },
    {
        "tickets_queue": [
            {"id": 5001, "title": "New employee onboarding", "description": "Set up accounts for 5 new hires starting Monday.", "status": "Open", "priority": 2, "created_by": "hr_ops"},
            {"id": 5002, "title": "VPN access request", "description": "Remote worker needs VPN credentials.", "status": "Resolved", "priority": 2, "created_by": "hr_ops"},
        ],
        "ticket_count": 5002, "ticket_counter": 5002,
        "current_user": "", "authenticated": False,
        "username": "hr_ops", "password": "testpass",
    },
    {
        "tickets_queue": [
            {"id": 9001, "title": "Payment gateway down", "description": "Credit card processing returning errors.", "status": "Open", "priority": 5, "created_by": "ops_lead"},
            {"id": 9002, "title": "SSL certificate expiring", "description": "Certificate for api.example.com expires in 3 days.", "status": "Open", "priority": 5, "created_by": "ops_lead"},
            {"id": 9003, "title": "CDN cache invalidation", "description": "Stale content being served to European users.", "status": "In Progress", "priority": 3, "created_by": "ops_lead"},
            {"id": 9004, "title": "Memory leak in worker", "description": "Background worker consuming 12GB after 24 hours.", "status": "Open", "priority": 4, "created_by": "ops_lead"},
            {"id": 9005, "title": "Backup job failing", "description": "Nightly database backup has failed for the last 2 nights.", "status": "Open", "priority": 4, "created_by": "ops_lead"},
        ],
        "ticket_count": 9005, "ticket_counter": 9005,
        "current_user": "", "authenticated": False,
        "username": "ops_lead", "password": "testpass",
    },
    {
        "tickets_queue": [],
        "ticket_count": 0, "ticket_counter": 0,
        "current_user": "", "authenticated": False,
        "username": "new_agent", "password": "testpass",
    },
]


TRAVEL_CONFIGS: List[Dict[str, Any]] = [
    {
        "credit_card_list": {
            "12345": {"card_number": "123456", "expiration_date": "12/2028", "cardholder_name": "Michael Smith", "card_verification_number": 465, "balance": 50000.0}
        },
        "booking_record": {
            "flight_001": {"travel_to": "Rome", "travel_from": "New York", "insurance": "none", "travel_cost": 1200.5, "travel_date": "2024-08-08", "travel_class": "Business", "transaction_id": "12345", "card_id": "12345"}
        },
        "access_token": "", "token_type": "Bearer", "token_expires_in": 0, "token_scope": "",
        "user_first_name": "Michael", "user_last_name": "Smith",
        "budget_limit": 5000.0,
        "client_id": "travel_client_001", "client_secret": "test_secret", "refresh_token": "refresh_abc123",
    },
    {
        "credit_card_list": {
            "22222": {"card_number": "222233", "expiration_date": "06/2026", "cardholder_name": "Sarah Johnson", "card_verification_number": 123, "balance": 15000.0}
        },
        "booking_record": {},
        "access_token": "", "token_type": "Bearer", "token_expires_in": 0, "token_scope": "",
        "user_first_name": "Sarah", "user_last_name": "Johnson",
        "budget_limit": 3000.0,
        "client_id": "biz_travel_002", "client_secret": "test_secret", "refresh_token": "refresh_def456",
    },
    {
        "credit_card_list": {
            "33333": {"card_number": "333344", "expiration_date": "09/2027", "cardholder_name": "Carlos Rivera", "card_verification_number": 789, "balance": 80000.0},
            "33334": {"card_number": "333355", "expiration_date": "03/2026", "cardholder_name": "Carlos Rivera", "card_verification_number": 321, "balance": 25000.0}
        },
        "booking_record": {
            "flight_010": {"travel_to": "Tokyo", "travel_from": "Los Angeles", "insurance": "full", "travel_cost": 2800.0, "travel_date": "2024-12-15", "travel_class": "First", "transaction_id": "33333", "card_id": "33333"}
        },
        "access_token": "", "token_type": "Bearer", "token_expires_in": 0, "token_scope": "",
        "user_first_name": "Carlos", "user_last_name": "Rivera",
        "budget_limit": 10000.0,
        "client_id": "lux_travel_003", "client_secret": "test_secret", "refresh_token": "refresh_ghi789",
    },
    {
        "credit_card_list": {
            "44444": {"card_number": "444455", "expiration_date": "01/2025", "cardholder_name": "Emily Chen", "card_verification_number": 456, "balance": 2000.0}
        },
        "booking_record": {},
        "access_token": "", "token_type": "Bearer", "token_expires_in": 0, "token_scope": "",
        "user_first_name": "Emily", "user_last_name": "Chen",
        "budget_limit": 1500.0,
        "client_id": "budget_004", "client_secret": "test_secret", "refresh_token": "refresh_jkl012",
    },
    {
        "credit_card_list": {
            "55555": {"card_number": "555566", "expiration_date": "11/2029", "cardholder_name": "David Park", "card_verification_number": 654, "balance": 120000.0}
        },
        "booking_record": {
            "flight_020": {"travel_to": "London", "travel_from": "San Francisco", "insurance": "full", "travel_cost": 3500.0, "travel_date": "2024-07-20", "travel_class": "Business", "transaction_id": "55555", "card_id": "55555"},
            "flight_021": {"travel_to": "Paris", "travel_from": "London", "insurance": "none", "travel_cost": 450.0, "travel_date": "2024-07-25", "travel_class": "Economy", "transaction_id": "55555", "card_id": "55555"},
        },
        "access_token": "", "token_type": "Bearer", "token_expires_in": 0, "token_scope": "",
        "user_first_name": "David", "user_last_name": "Park",
        "budget_limit": 8000.0,
        "client_id": "corp_travel_005", "client_secret": "test_secret", "refresh_token": "refresh_mno345",
    },
]


POSTING_CONFIGS: List[Dict[str, Any]] = [
    {
        "authenticated": False, "tweet_counter": 17,
        "tweets": {
            "0": {"id": 0, "username": "genealogy_enthusiast", "content": "Excited to start my genealogy journey!", "tags": ["#genealogy", "#familyhistory", "#beginnings"], "mentions": []},
            "1": {"id": 1, "username": "genealogy_enthusiast", "content": "Researching family history is so rewarding.", "tags": ["#genealogy", "#research", "#familyhistory"], "mentions": []},
            "2": {"id": 2, "username": "genealogy_enthusiast", "content": "Can't wait to uncover new stories about my ancestors.", "tags": ["#ancestors", "#familystories", "#discovery"], "mentions": []},
            "3": {"id": 3, "username": "genealogy_enthusiast", "content": "Genealogy is like a puzzle waiting to be solved.", "tags": ["#genealogy", "#puzzle", "#research"], "mentions": []},
            "4": {"id": 4, "username": "genealogy_enthusiast", "content": "Every family has a story worth telling.", "tags": ["#familystories", "#heritage", "#genealogy"], "mentions": []},
            "5": {"id": 5, "username": "genealogy_enthusiast", "content": "Exploring my roots is a journey of self-discovery.", "tags": ["#roots", "#selfdiscovery", "#familyhistory"], "mentions": []},
            "6": {"id": 6, "username": "genealogy_enthusiast", "content": "Family history is a treasure trove of stories.", "tags": ["#familyhistory", "#stories", "#heritage"], "mentions": []},
            "7": {"id": 7, "username": "genealogy_enthusiast", "content": "Connecting with my past to understand my present.", "tags": ["#connection", "#pastpresent", "#genealogy"], "mentions": []},
            "8": {"id": 8, "username": "genealogy_enthusiast", "content": "Genealogy: where history meets personal stories.", "tags": ["#genealogy", "#history", "#personalstories"], "mentions": []},
            "9": {"id": 9, "username": "genealogy_enthusiast", "content": "Uncovering the past, one ancestor at a time.", "tags": ["#ancestors", "#research", "#familyhistory"], "mentions": []},
            "10": {"id": 10, "username": "tech_user", "content": "Artificial intelligence is transforming how we work and live.", "tags": ["#AI", "#technology", "#future"], "mentions": []},
            "11": {"id": 11, "username": "tech_user", "content": "Just published a new blog post about machine learning trends.", "tags": ["#machinelearning", "#AI", "#blog"], "mentions": []},
            "12": {"id": 12, "username": "tech_user", "content": "Exploring the latest breakthroughs in natural language processing.", "tags": ["#NLP", "#AI", "#research"], "mentions": []},
            "13": {"id": 13, "username": "tech_user", "content": "The future of artificial intelligence is bright and full of possibilities.", "tags": ["#AI", "#future", "#technology"], "mentions": []},
            "14": {"id": 14, "username": "history_buff", "content": "Ancient civilizations had remarkable engineering skills.", "tags": ["#history", "#engineering", "#ancient"], "mentions": []},
            "15": {"id": 15, "username": "family_finder", "content": "Found a new branch of my family tree today!", "tags": ["#familytree", "#genealogy", "#discovery"], "mentions": []},
            "16": {"id": 16, "username": "puzzle_solver", "content": "AI-powered tools make data analysis so much easier.", "tags": ["#AI", "#data", "#analytics"], "mentions": []},
        },
        "comments": {
            "0": [{"username": "history_buff", "content": "Great start!"}, {"username": "family_finder", "content": "Welcome to the community!"}],
            "3": [{"username": "puzzle_solver", "content": "So true!"}],
        },
        "retweets": [{"username": "history_buff", "tweet_id": 0}, {"username": "family_finder", "tweet_id": 5}],
        "following_list": ["history_buff", "family_finder", "puzzle_solver"],
        "users": {"history_buff": {"tweet_count": 25, "following_count": 50, "retweet_count": 10}, "family_finder": {"tweet_count": 15, "following_count": 30, "retweet_count": 5}, "puzzle_solver": {"tweet_count": 8, "following_count": 12, "retweet_count": 2}},
        "username": "tech_user", "password": "testpass",
    },
    {
        "authenticated": False, "tweet_counter": 5,
        "tweets": {
            "0": {"id": 0, "username": "foodie_chef", "content": "Just perfected my sourdough recipe!", "tags": ["#baking", "#sourdough", "#homemade"], "mentions": []},
            "1": {"id": 1, "username": "foodie_chef", "content": "Farm-to-table dinner tonight. Fresh ingredients make all the difference.", "tags": ["#farmtotable", "#dinner", "#fresh"], "mentions": []},
            "2": {"id": 2, "username": "foodie_chef", "content": "Best ramen spot in town? Drop your recommendations below!", "tags": ["#ramen", "#foodie", "#recommendations"], "mentions": []},
            "3": {"id": 3, "username": "food_critic", "content": "That new fusion restaurant exceeded expectations.", "tags": ["#restaurant", "#fusion", "#review"], "mentions": []},
            "4": {"id": 4, "username": "home_baker", "content": "First attempt at croissants - not bad!", "tags": ["#baking", "#croissants", "#firsttry"], "mentions": []},
            "5": {"id": 5, "username": "foodie_chef", "content": "Sunday meal prep done. Ready for the week!", "tags": ["#mealprep", "#cooking", "#sunday"], "mentions": []},
        },
        "comments": {"2": [{"username": "food_critic", "content": "Try the place on 5th Ave!"}], "4": [{"username": "foodie_chef", "content": "Keep practicing, they look great!"}]},
        "retweets": [{"username": "food_critic", "tweet_id": 0}],
        "following_list": ["food_critic", "home_baker"],
        "users": {"food_critic": {"tweet_count": 40, "following_count": 80, "retweet_count": 15}, "home_baker": {"tweet_count": 12, "following_count": 25, "retweet_count": 3}},
        "username": "foodie_chef", "password": "testpass",
    },
    {
        "authenticated": False, "tweet_counter": 3,
        "tweets": {
            "0": {"id": 0, "username": "travel_diaries", "content": "Just landed in Tokyo! First stop: Shibuya crossing.", "tags": ["#travel", "#tokyo", "#japan"], "mentions": []},
            "1": {"id": 1, "username": "travel_diaries", "content": "Sunrise at Angkor Wat - breathtaking.", "tags": ["#cambodia", "#sunrise", "#wanderlust"], "mentions": []},
            "2": {"id": 2, "username": "globe_trotter", "content": "Top 10 hidden gems in Portugal thread coming soon.", "tags": ["#portugal", "#travel", "#hiddengems"], "mentions": []},
            "3": {"id": 3, "username": "travel_diaries", "content": "Street food in Bangkok is unbeatable.", "tags": ["#bangkok", "#streetfood", "#thailand"], "mentions": []},
        },
        "comments": {"0": [{"username": "globe_trotter", "content": "Enjoy! Try the ramen in Shinjuku."}]},
        "retweets": [{"username": "globe_trotter", "tweet_id": 1}],
        "following_list": ["globe_trotter", "backpacker_joe"],
        "users": {"globe_trotter": {"tweet_count": 55, "following_count": 120, "retweet_count": 30}, "backpacker_joe": {"tweet_count": 20, "following_count": 45, "retweet_count": 8}},
        "username": "travel_diaries", "password": "testpass",
    },
    {
        "authenticated": False, "tweet_counter": 25,
        "tweets": {
            "0": {"id": 0, "username": "techguru", "content": "Just launched my new coding tutorial series!", "tags": ["#coding", "#tutorial", "#programming"], "mentions": []},
            "1": {"id": 1, "username": "techguru", "content": "Debugging tips: always check your types first.", "tags": ["#debugging", "#coding"], "mentions": []},
            "2": {"id": 2, "username": "foodie_chef", "content": "Made homemade pasta from scratch today!", "tags": ["#cooking", "#pasta", "#homemade"], "mentions": []},
            "3": {"id": 3, "username": "foodie_chef", "content": "The secret to good soup is patience.", "tags": ["#cooking", "#tips", "#soup"], "mentions": []},
            "4": {"id": 4, "username": "travel_lover", "content": "Bangkok street food tour was amazing!", "tags": ["#travel", "#food", "#bangkok"], "mentions": []},
            "5": {"id": 5, "username": "travel_lover", "content": "Sunrise at Angkor Wat - unforgettable!", "tags": ["#travel", "#sunrise", "#cambodia"], "mentions": []},
            "6": {"id": 6, "username": "fitness_junkie", "content": "Morning run complete - 5K in 25 minutes!", "tags": ["#fitness", "#running", "#morning"], "mentions": []},
            "7": {"id": 7, "username": "fitness_junkie", "content": "Protein shake recipe that actually tastes good.", "tags": ["#fitness", "#nutrition", "#recipe"], "mentions": []},
        },
        "comments": {
            "2": [{"username": "foodie_chef", "content": "Enjoy the pasta journey!"}],
        },
        "retweets": [],
        "following_list": [],
        "users": {
            "techguru": {"tweet_count": 45, "following_count": 30, "retweet_count": 12},
            "foodie_chef": {"tweet_count": 38, "following_count": 25, "retweet_count": 8},
            "travel_lover": {"tweet_count": 52, "following_count": 40, "retweet_count": 15},
            "fitness_junkie": {"tweet_count": 30, "following_count": 20, "retweet_count": 5},
        },
        "username": "newcomer", "password": "testpass",
    },
    {
        "authenticated": False, "tweet_counter": 25,
        "tweets": {
            "0": {"id": 0, "username": "sports_fan", "content": "What a game last night!", "tags": ["#sports", "#basketball", "#NBA"], "mentions": []},
            "1": {"id": 1, "username": "sports_fan", "content": "Predictions for this weekend's matches?", "tags": ["#predictions", "#weekend", "#sports"], "mentions": []},
            "2": {"id": 2, "username": "sports_fan", "content": "That trade was unexpected.", "tags": ["#trade", "#NBA", "#breaking"], "mentions": []},
            "3": {"id": 3, "username": "sports_fan", "content": "MVP race is heating up this season.", "tags": ["#MVP", "#NBA", "#season"], "mentions": []},
            "4": {"id": 4, "username": "sports_fan", "content": "Rookie watch: top 5 performers this week.", "tags": ["#rookie", "#NBA", "#top5"], "mentions": []},
            "5": {"id": 5, "username": "sports_analyst", "content": "Breaking down the defensive strategies of the top 3 teams.", "tags": ["#analysis", "#defense", "#basketball"], "mentions": []},
            "6": {"id": 6, "username": "sports_analyst", "content": "Advanced stats show this player is underrated.", "tags": ["#analytics", "#NBA", "#underrated"], "mentions": []},
            "7": {"id": 7, "username": "sports_fan", "content": "Hall of Fame debate: who gets in this year?", "tags": ["#HallOfFame", "#debate", "#NBA"], "mentions": []},
            "8": {"id": 8, "username": "sports_fan", "content": "Best dunks of the season compilation.", "tags": ["#dunks", "#highlights", "#NBA"], "mentions": []},
            "9": {"id": 9, "username": "sports_fan", "content": "Fantasy league update: trade deadline approaching.", "tags": ["#fantasy", "#basketball", "#trade"], "mentions": []},
            "10": {"id": 10, "username": "sports_analyst", "content": "Power rankings after this week's games.", "tags": ["#powerrankings", "#NBA", "#weekly"], "mentions": []},
            "11": {"id": 11, "username": "sports_fan", "content": "Coaching changes rumoured for next season.", "tags": ["#coaching", "#rumors", "#NBA"], "mentions": []},
            "12": {"id": 12, "username": "sports_fan", "content": "Playoff picture is taking shape!", "tags": ["#playoffs", "#NBA", "#postseason"], "mentions": []},
            "13": {"id": 13, "username": "sports_fan", "content": "Injury report: key players to watch.", "tags": ["#injury", "#NBA", "#health"], "mentions": []},
            "14": {"id": 14, "username": "sports_fan", "content": "Draft prospect spotlight.", "tags": ["#draft", "#prospects", "#NBA"], "mentions": []},
            "15": {"id": 15, "username": "sports_fan", "content": "All-Star voting is open!", "tags": ["#AllStar", "#vote", "#NBA"], "mentions": []},
            "16": {"id": 16, "username": "sports_analyst", "content": "Comparing eras: 90s vs today's game.", "tags": ["#debate", "#NBA", "#history"], "mentions": []},
            "17": {"id": 17, "username": "sports_fan", "content": "Free agency predictions.", "tags": ["#freeagency", "#NBA", "#predictions"], "mentions": []},
            "18": {"id": 18, "username": "sports_fan", "content": "Top 10 plays of the month.", "tags": ["#top10", "#plays", "#NBA"], "mentions": []},
            "19": {"id": 19, "username": "sports_fan", "content": "International players making waves.", "tags": ["#international", "#NBA", "#global"], "mentions": []},
            "20": {"id": 20, "username": "sports_fan", "content": "Chemistry matters: team dynamics this season.", "tags": ["#team", "#chemistry", "#NBA"], "mentions": []},
            "21": {"id": 21, "username": "sports_analyst", "content": "Salary cap implications for next year.", "tags": ["#salarycap", "#NBA", "#business"], "mentions": []},
            "22": {"id": 22, "username": "sports_fan", "content": "Bench mob appreciation post.", "tags": ["#bench", "#NBA", "#appreciation"], "mentions": []},
            "23": {"id": 23, "username": "sports_fan", "content": "Game thread: tonights matchup.", "tags": ["#gamethread", "#NBA", "#live"], "mentions": []},
            "24": {"id": 24, "username": "sports_fan", "content": "Season ticket holder perks this year.", "tags": ["#tickets", "#NBA", "#perks"], "mentions": []},
            "25": {"id": 25, "username": "sports_fan", "content": "End of season awards predictions.", "tags": ["#awards", "#NBA", "#predictions"], "mentions": []},
        },
        "comments": {
            "0": [{"username": "sports_analyst", "content": "Incredible finish!"}],
            "3": [{"username": "sports_analyst", "content": "My money is on player X."}],
            "12": [{"username": "sports_analyst", "content": "West is stacked this year."}],
        },
        "retweets": [{"username": "sports_analyst", "tweet_id": 0}, {"username": "sports_analyst", "tweet_id": 5}],
        "following_list": ["sports_analyst", "stat_guru"],
        "users": {"sports_analyst": {"tweet_count": 100, "following_count": 200, "retweet_count": 45}, "stat_guru": {"tweet_count": 30, "following_count": 50, "retweet_count": 10}},
        "username": "sports_fan", "password": "testpass",
    },
]


MESSAGE_CONFIGS: List[Dict[str, Any]] = [
    {
        "workspace_id": "WS123456", "user_count": 10,
        "user_map": {
            "Michael": "USR005", "Sarah": "USR006", "David": "USR007", "Emma": "USR008",
            "Alice": "USR009", "Bob": "USR010", "Charlie": "USR011", "Diana": "USR012",
            "John": "USR013", "Jane": "USR014",
        },
        "messages_sent_map": {
            "USR005": {"USR006": [{"message_id": 1, "message": "Please review the attached document."}], "USR007": [{"message_id": 2, "message": "Meeting at 3 PM."}], "USR008": [{"message_id": 3, "message": "Lunch tomorrow?"}]},
            "USR006": {"USR005": [{"message_id": 4, "message": "Got it, thanks!"}], "USR007": [{"message_id": 5, "message": "Can we reschedule?"}]},
            "USR007": {"USR005": [{"message_id": 6, "message": "Sure, see you then."}], "USR008": [{"message_id": 7, "message": "Let's catch up soon."}]},
            "USR008": {"USR006": [{"message_id": 8, "message": "I'll be there."}], "USR007": [{"message_id": 9, "message": "Sounds good."}]},
            "USR009": {}, "USR010": {}, "USR011": {}, "USR012": {}, "USR013": {}, "USR014": {},
        },
        "messages_inbox_map": {
            "USR005": {"USR006": [{"message_id": 4, "message": "Got it, thanks!"}], "USR007": [{"message_id": 6, "message": "Sure, see you then."}]},
            "USR006": {"USR005": [{"message_id": 1, "message": "Please review the attached document."}], "USR008": [{"message_id": 8, "message": "I'll be there."}]},
            "USR007": {"USR005": [{"message_id": 2, "message": "Meeting at 3 PM."}], "USR006": [{"message_id": 5, "message": "Can we reschedule?"}]},
            "USR008": {"USR005": [{"message_id": 3, "message": "Lunch tomorrow?"}], "USR007": [{"message_id": 7, "message": "Let's catch up soon."}]},
            "USR009": {}, "USR010": {}, "USR011": {}, "USR012": {}, "USR013": {}, "USR014": {},
        },
        "message_count": 9, "current_user": "",
    },
    {
        "workspace_id": "WS789012", "user_count": 4,
        "user_map": {"Tom": "USR101", "Lisa": "USR102", "Nina": "USR103", "Raj": "USR104"},
        "messages_sent_map": {
            "USR101": {"USR102": [{"message_id": 1, "message": "Can you review the PR?"}, {"message_id": 2, "message": "Updated the branch."}], "USR103": [{"message_id": 3, "message": "Standup at 10."}]},
            "USR102": {"USR101": [{"message_id": 4, "message": "Looks good, approved."}]},
            "USR103": {"USR104": [{"message_id": 5, "message": "Can you deploy to staging?"}]},
            "USR104": {},
        },
        "messages_inbox_map": {
            "USR101": {"USR102": [{"message_id": 4, "message": "Looks good, approved."}]},
            "USR102": {"USR101": [{"message_id": 1, "message": "Can you review the PR?"}, {"message_id": 2, "message": "Updated the branch."}]},
            "USR103": {"USR101": [{"message_id": 3, "message": "Standup at 10."}]},
            "USR104": {"USR103": [{"message_id": 5, "message": "Can you deploy to staging?"}]},
        },
        "message_count": 5, "current_user": "",
    },
    {
        "workspace_id": "WS_CLASS_001", "user_count": 6,
        "user_map": {"Prof Adams": "USR201", "Maria": "USR202", "James": "USR203", "Yuki": "USR204", "Ahmed": "USR205", "Sofia": "USR206"},
        "messages_sent_map": {
            "USR201": {"USR202": [{"message_id": 1, "message": "Assignment deadline extended to Friday."}], "USR203": [{"message_id": 2, "message": "Please submit your project."}]},
            "USR202": {"USR203": [{"message_id": 3, "message": "Want to study together?"}], "USR204": [{"message_id": 4, "message": "Did you finish the reading?"}]},
            "USR203": {"USR201": [{"message_id": 5, "message": "Will submit by tonight."}]},
            "USR204": {"USR202": [{"message_id": 6, "message": "Yes, chapters 5-8."}]},
            "USR205": {},
            "USR206": {},
        },
        "messages_inbox_map": {
            "USR201": {"USR203": [{"message_id": 5, "message": "Will submit by tonight."}]},
            "USR202": {"USR201": [{"message_id": 1, "message": "Assignment deadline extended to Friday."}], "USR204": [{"message_id": 7, "message": "Did you finish the reading?"}]},
            "USR203": {"USR201": [{"message_id": 2, "message": "Please submit your project."}], "USR202": [{"message_id": 3, "message": "Want to study together?"}]},
            "USR204": {"USR202": [{"message_id": 8, "message": "Yes, chapters 5-8."}]},
            "USR205": {},
            "USR206": {},
        },
        "message_count": 8, "current_user": "",
    },
    {
        "workspace_id": "WS_FAMILY_001", "user_count": 5,
        "user_map": {"Mom": "USR301", "Dad": "USR302", "Alex": "USR303", "Sam": "USR304", "Grandma": "USR305"},
        "messages_sent_map": {
            "USR301": {"USR303": [{"message_id": 1, "message": "Don't forget dinner at 7."}], "USR304": [{"message_id": 2, "message": "Bring your jacket."}]},
            "USR302": {"USR303": [{"message_id": 3, "message": "Can you help with the computer?"}]},
            "USR303": {"USR301": [{"message_id": 4, "message": "I'll be there!"}]},
            "USR304": {"USR301": [{"message_id": 5, "message": "Yes mom."}]},
            "USR305": {},
        },
        "messages_inbox_map": {
            "USR301": {"USR303": [{"message_id": 4, "message": "I'll be there!"}], "USR304": [{"message_id": 5, "message": "Yes mom."}]},
            "USR302": {},
            "USR303": {"USR301": [{"message_id": 1, "message": "Don't forget dinner at 7."}], "USR302": [{"message_id": 3, "message": "Can you help with the computer?"}]},
            "USR304": {"USR301": [{"message_id": 2, "message": "Bring your jacket."}]},
            "USR305": {},
        },
        "message_count": 5, "current_user": "",
    },
    {
        "workspace_id": "WS_STARTUP_001", "user_count": 3,
        "user_map": {"Founder": "USR401", "CTO": "USR402", "Designer": "USR403"},
        "messages_sent_map": {
            "USR401": {"USR402": [{"message_id": 1, "message": "Investor meeting tomorrow."}], "USR403": [{"message_id": 2, "message": "New mockups needed."}]},
            "USR402": {"USR401": [{"message_id": 3, "message": "API is ready for demo."}]},
            "USR403": {"USR402": [{"message_id": 4, "message": "Design specs are in Figma."}]},
        },
        "messages_inbox_map": {
            "USR401": {"USR402": [{"message_id": 3, "message": "API is ready for demo."}]},
            "USR402": {"USR401": [{"message_id": 1, "message": "Investor meeting tomorrow."}], "USR403": [{"message_id": 4, "message": "Design specs are in Figma."}]},
            "USR403": {"USR401": [{"message_id": 2, "message": "New mockups needed."}]},
        },
        "message_count": 4, "current_user": "",
    },
]


FILESYSTEM_CONFIGS: List[Dict[str, Any]] = [
    {
        "root": {
            "readme.txt": {"type": "file", "content": "Welcome to the workspace.\nThis is a shared document area.\nPlease follow the naming conventions."},
            "report.csv": {"type": "file", "content": "name,score,grade\nAlice,92,A\nBob,78,B"},
            "documents": {"type": "directory", "contents": {}},
            "projects": {"type": "directory", "contents": {}},
        }
    },
    {
        "root": {
            "main.py": {"type": "file", "content": "def main():\n    print('Hello, World!')\n\nif __name__ == '__main__':\n    main()"},
            "utils.py": {"type": "file", "content": "import json\n\ndef parse():\n    pass"},
            "users.json": {"type": "file", "content": '[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]'},
            "config.yaml": {"type": "file", "content": "server:\n  port: 8080\n  host: localhost"},
            "code": {"type": "directory", "contents": {}},
            "data": {"type": "directory", "contents": {}},
        }
    },
    {
        "root": {
            "beach.jpg": {"type": "file", "content": "[binary: beach photo 2.4MB]"},
            "sunset.jpg": {"type": "file", "content": "[binary: sunset photo 1.8MB]"},
            "todo.txt": {"type": "file", "content": "1. Renew passport\n2. Schedule dentist"},
            "photos": {"type": "directory", "contents": {}},
            "documents": {"type": "directory", "contents": {}},
        }
    },
    {
        "root": {
            "access.log": {"type": "file", "content": '192.168.1.1 - - [01/Jan/2024:10:00:00] "GET / HTTP/1.1" 200\n192.168.1.2 - - [01/Jan/2024:10:01:00] "POST /api HTTP/1.1" 201'},
            "error.log": {"type": "file", "content": "[ERROR] 2024-01-01 10:05:00 - Connection timeout\n[ERROR] 2024-01-01 10:10:00 - Disk space low"},
            "backup.sh": {"type": "file", "content": "#!/bin/bash\necho 'Backup complete'"},
            "monitor.py": {"type": "file", "content": "import psutil\n\ndef check():\n    return True"},
            "server": {"type": "directory", "contents": {}},
            "scripts": {"type": "directory", "contents": {}},
        }
    },
    {
        "root": {
            "paper_draft.txt": {"type": "file", "content": "Abstract: This paper explores...\n\n1. Introduction\n2. Methods\n3. Results"},
            "references.bib": {"type": "file", "content": '@article{smith2024,\n  title={Deep Learning},\n  year={2024}\n}'},
            "meeting_notes.txt": {"type": "file", "content": "Meeting with advisor\n- Review draft by Friday"},
            "research": {"type": "directory", "contents": {}},
            "notes": {"type": "directory", "contents": {}},
        }
    },
]


PERSONAS: List[Dict[str, str]] = [
    {"name": "Akira Tanaka", "city": "Tokyo", "country": "Japan"},
    {"name": "Yuki Sato", "city": "Osaka", "country": "Japan"},
    {"name": "Maria Garcia", "city": "Barcelona", "country": "Spain"},
    {"name": "Carlos Rodriguez", "city": "Madrid", "country": "Spain"},
    {"name": "Chen Wei", "city": "Shanghai", "country": "China"},
    {"name": "Liu Mei", "city": "Beijing", "country": "China"},
    {"name": "Priya Sharma", "city": "Mumbai", "country": "India"},
    {"name": "Raj Patel", "city": "Delhi", "country": "India"},
    {"name": "Lars Eriksson", "city": "Stockholm", "country": "Sweden"},
    {"name": "Ingrid Berg", "city": "Oslo", "country": "Norway"},
    {"name": "Fatima Al-Rashid", "city": "Dubai", "country": "UAE"},
    {"name": "Omar Hassan", "city": "Cairo", "country": "Egypt"},
    {"name": "Sofia Rossi", "city": "Rome", "country": "Italy"},
    {"name": "Marco Bianchi", "city": "Milan", "country": "Italy"},
    {"name": "Anna Kowalski", "city": "Warsaw", "country": "Poland"},
    {"name": "Jan Novak", "city": "Prague", "country": "Czech Republic"},
    {"name": "Hans Mueller", "city": "Berlin", "country": "Germany"},
    {"name": "Petra Schmidt", "city": "Munich", "country": "Germany"},
    {"name": "Jean Dupont", "city": "Lyon", "country": "France"},
    {"name": "Claire Martin", "city": "Paris", "country": "France"},
    {"name": "James Wilson", "city": "London", "country": "UK"},
    {"name": "Emily Taylor", "city": "Manchester", "country": "UK"},
    {"name": "Sarah O'Brien", "city": "Dublin", "country": "Ireland"},
    {"name": "Liam O'Connor", "city": "Cork", "country": "Ireland"},
    {"name": "Mikhail Petrov", "city": "Moscow", "country": "Russia"},
    {"name": "Elena Ivanova", "city": "St. Petersburg", "country": "Russia"},
    {"name": "Lucas Silva", "city": "Sao Paulo", "country": "Brazil"},
    {"name": "Ana Santos", "city": "Rio de Janeiro", "country": "Brazil"},
    {"name": "Diego Martinez", "city": "Mexico City", "country": "Mexico"},
    {"name": "Carmen Lopez", "city": "Guadalajara", "country": "Mexico"},
    {"name": "Joon-ho Kim", "city": "Seoul", "country": "South Korea"},
    {"name": "Min-ji Park", "city": "Busan", "country": "South Korea"},
    {"name": "Tomoko Abe", "city": "Nagoya", "country": "Japan"},
    {"name": "Kenji Yamamoto", "city": "Fukuoka", "country": "Japan"},
    {"name": "Aisha Khan", "city": "Lahore", "country": "Pakistan"},
    {"name": "Ravi Nair", "city": "Bangalore", "country": "India"},
    {"name": "David Cohen", "city": "Tel Aviv", "country": "Israel"},
    {"name": "Sarah Levy", "city": "Jerusalem", "country": "Israel"},
    {"name": "Ahmet Yilmaz", "city": "Istanbul", "country": "Turkey"},
    {"name": "Elif Demir", "city": "Ankara", "country": "Turkey"},
    {"name": "Olga Kovalenko", "city": "Kyiv", "country": "Ukraine"},
    {"name": "Andrei Popov", "city": "Sofia", "country": "Bulgaria"},
    {"name": "Ruth Nwosu", "city": "Lagos", "country": "Nigeria"},
    {"name": "Kwame Asante", "city": "Accra", "country": "Ghana"},
    {"name": "Thabo Mokoena", "city": "Johannesburg", "country": "South Africa"},
    {"name": "Amara Diallo", "city": "Dakar", "country": "Senegal"},
    {"name": "Liam Campbell", "city": "Toronto", "country": "Canada"},
    {"name": "Marie Tremblay", "city": "Montreal", "country": "Canada"},
    {"name": "Robert Brown", "city": "Sydney", "country": "Australia"},
    {"name": "Emma Watson", "city": "Melbourne", "country": "Australia"},
    {"name": "Jack Zhang", "city": "Auckland", "country": "New Zealand"},
    {"name": "Sophie Chen", "city": "Vancouver", "country": "Canada"},
]

CITIES: List[Dict[str, str]] = [
    {"city": "New York", "state": "NY", "code": "JFK"},
    {"city": "Los Angeles", "state": "CA", "code": "LAX"},
    {"city": "San Francisco", "state": "CA", "code": "SFO"},
    {"city": "Chicago", "state": "IL", "code": "ORD"},
    {"city": "Boston", "state": "MA", "code": "BOS"},
    {"city": "Seattle", "state": "WA", "code": "SEA"},
    {"city": "Miami", "state": "FL", "code": "MIA"},
    {"city": "Denver", "state": "CO", "code": "DEN"},
    {"city": "Atlanta", "state": "GA", "code": "ATL"},
    {"city": "Dallas", "state": "TX", "code": "DFW"},
    {"city": "Portland", "state": "OR", "code": "PDX"},
    {"city": "Phoenix", "state": "AZ", "code": "PHX"},
    {"city": "Detroit", "state": "MI", "code": "DTW"},
    {"city": "Minneapolis", "state": "MN", "code": "MSP"},
    {"city": "Houston", "state": "TX", "code": "IAH"},
    {"city": "Philadelphia", "state": "PA", "code": "PHL"},
    {"city": "San Diego", "state": "CA", "code": "SAN"},
    {"city": "Washington", "state": "DC", "code": "DCA"},
    {"city": "London", "state": "", "code": "LHR"},
    {"city": "Paris", "state": "", "code": "CDG"},
    {"city": "Tokyo", "state": "", "code": "NRT"},
    {"city": "Berlin", "state": "", "code": "BER"},
    {"city": "Rome", "state": "", "code": "FCO"},
    {"city": "Madrid", "state": "", "code": "MAD"},
    {"city": "Barcelona", "state": "", "code": "BCN"},
    {"city": "Amsterdam", "state": "", "code": "AMS"},
    {"city": "Istanbul", "state": "", "code": "IST"},
    {"city": "Dubai", "state": "", "code": "DXB"},
    {"city": "Singapore", "state": "", "code": "SIN"},
    {"city": "Hong Kong", "state": "", "code": "HKG"},
    {"city": "Sydney", "state": "", "code": "SYD"},
    {"city": "Seoul", "state": "", "code": "ICN"},
    {"city": "Mumbai", "state": "", "code": "BOM"},
    {"city": "Sao Paulo", "state": "", "code": "GRU"},
    {"city": "Mexico City", "state": "", "code": "MEX"},
    {"city": "Toronto", "state": "ON", "code": "YYZ"},
    {"city": "Montreal", "state": "QC", "code": "YUL"},
    {"city": "Shanghai", "state": "", "code": "PVG"},
    {"city": "Bangkok", "state": "", "code": "BKK"},
    {"city": "Cairo", "state": "", "code": "CAI"},
    {"city": "Nairobi", "state": "", "code": "NBO"},
    {"city": "Johannesburg", "state": "", "code": "JNB"},
    {"city": "Buenos Aires", "state": "", "code": "EZE"},
    {"city": "Lima", "state": "", "code": "LIM"},
    {"city": "Bogota", "state": "", "code": "BOG"},
    {"city": "Santiago", "state": "", "code": "SCL"},
    {"city": "Lisbon", "state": "", "code": "LIS"},
    {"city": "Vienna", "state": "", "code": "VIE"},
    {"city": "Prague", "state": "", "code": "PRG"},
    {"city": "Warsaw", "state": "", "code": "WAW"},
]


def generate_query_seed(rng: random.Random = None) -> dict:
    """Pick a random persona and city for query/argument diversity.

    Args:
        rng: Optional Random instance. None = module-level random.
    """
    choice = rng.choice if rng else random.choice
    return {"persona": choice(PERSONAS), "city": choice(CITIES)}


def generate_random_config(seed: int = None) -> Dict[str, Any]:
    """Pick one variation per domain and combine into a full config dict.

    Args:
        seed: Optional RNG seed for reproducibility.  None = random each call.

    Returns:
        A dict with keys matching FULL_INITIAL_CONFIGS structure.
    """
    rng = random.Random(seed)

    config = {
        "GorillaFileSystem": copy.deepcopy(rng.choice(FILESYSTEM_CONFIGS)),
        "MathAPI": {"numbers": [275.5, 299.75, 250.65, 310.85, 290.1]},
        "MessageAPI": copy.deepcopy(rng.choice(MESSAGE_CONFIGS)),
        "PostingAPI": copy.deepcopy(rng.choice(POSTING_CONFIGS)),
        "TicketAPI": copy.deepcopy(rng.choice(TICKET_CONFIGS)),
        "TradingBot": copy.deepcopy(rng.choice(TRADING_CONFIGS)),
        "TravelAPI": copy.deepcopy(rng.choice(TRAVEL_CONFIGS)),
        "VehicleControlAPI": copy.deepcopy(rng.choice(VEHICLE_CONFIGS)),
    }

    return config
