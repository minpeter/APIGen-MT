"""Declarative tool catalog and canonical initial-state data."""

from __future__ import annotations

import copy
import importlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence

if __package__:
    from .tool_manager_types import (
        Config,
        ConfigMap,
        ToolSchema,
        get_attribute,
        is_string_object_dict,
        require_config_list,
        string_field,
    )
else:
    from tool_manager_types import (
        Config,
        ConfigMap,
        ToolSchema,
        get_attribute,
        is_string_object_dict,
        require_config_list,
        string_field,
    )

_SIBLING_PREFIX = f"{__package__}." if __package__ else ""
_config_pool = importlib.import_module(f"{_SIBLING_PREFIX}config_pool")
POSTING_CONFIGS = require_config_list(
    get_attribute(_config_pool, "POSTING_CONFIGS"), source="POSTING_CONFIGS"
)
TICKET_CONFIGS = require_config_list(
    get_attribute(_config_pool, "TICKET_CONFIGS"), source="TICKET_CONFIGS"
)
TRADING_CONFIGS = require_config_list(
    get_attribute(_config_pool, "TRADING_CONFIGS"), source="TRADING_CONFIGS"
)
TRAVEL_CONFIGS = require_config_list(
    get_attribute(_config_pool, "TRAVEL_CONFIGS"), source="TRAVEL_CONFIGS"
)
VEHICLE_CONFIGS = require_config_list(
    get_attribute(_config_pool, "VEHICLE_CONFIGS"), source="VEHICLE_CONFIGS"
)


CLASS_KEY_TO_INITIAL_CONFIG_KEY = {
    "gorilla_file_system": "GorillaFileSystem",
    "math_api": "MathAPI",
    "message_api": "MessageAPI",
    "posting_api": "PostingAPI",
    "ticket_api": "TicketAPI",
    "trading_bot": "TradingBot",
    "travel_booking": "TravelAPI",
    "vehicle_control": "VehicleControlAPI",
}

CLASS_KEY_TO_CLASS_NAME = {
    "gorilla_file_system": "GorillaFileSystem",
    "math_api": "MathAPI",
    "message_api": "MessageAPI",
    "posting_api": "PostingAPI",
    "ticket_api": "TicketAPI",
    "trading_bot": "TradingBot",
    "travel_booking": "TravelBooking",
    "vehicle_control": "VehicleControl",
}

TOOL_CLASS_KEYS = list(CLASS_KEY_TO_CLASS_NAME.keys())

TOOL_NAME_TO_CLASS_KEY = {
    "cat": "gorilla_file_system",
    "cd": "gorilla_file_system",
    "cp": "gorilla_file_system",
    "diff": "gorilla_file_system",
    "du": "gorilla_file_system",
    "echo": "gorilla_file_system",
    "find": "gorilla_file_system",
    "grep": "gorilla_file_system",
    "ls": "gorilla_file_system",
    "mkdir": "gorilla_file_system",
    "mv": "gorilla_file_system",
    "rm": "gorilla_file_system",
    "rmdir": "gorilla_file_system",
    "sort": "gorilla_file_system",
    "tail": "gorilla_file_system",
    "touch": "gorilla_file_system",
    "wc": "gorilla_file_system",
    "absolute_value": "math_api",
    "add": "math_api",
    "divide": "math_api",
    "imperial_si_conversion": "math_api",
    "logarithm": "math_api",
    "max_value": "math_api",
    "mean": "math_api",
    "min_value": "math_api",
    "multiply": "math_api",
    "percentage": "math_api",
    "power": "math_api",
    "round_number": "math_api",
    "si_unit_conversion": "math_api",
    "square_root": "math_api",
    "standard_deviation": "math_api",
    "subtract": "math_api",
    "sum_values": "math_api",
    "add_contact": "message_api",
    "delete_message": "message_api",
    "get_user_id": "message_api",
    "message_login": "message_api",
    "search_messages": "message_api",
    "send_message": "message_api",
    "authenticate_twitter": "posting_api",
    "comment": "posting_api",
    "follow_user": "posting_api",
    "get_tweet": "posting_api",
    "get_tweet_comments": "posting_api",
    "get_user_stats": "posting_api",
    "get_user_tweets": "posting_api",
    "mention": "posting_api",
    "post_tweet": "posting_api",
    "retweet": "posting_api",
    "search_tweets": "posting_api",
    "unfollow_user": "posting_api",
    "close_ticket": "ticket_api",
    "create_ticket": "ticket_api",
    "edit_ticket": "ticket_api",
    "get_ticket": "ticket_api",
    "get_user_tickets": "ticket_api",
    "resolve_ticket": "ticket_api",
    "ticket_login": "ticket_api",
    "add_to_watchlist": "trading_bot",
    "cancel_order": "trading_bot",
    "filter_stocks_by_price": "trading_bot",
    "fund_account": "trading_bot",
    "get_available_stocks": "trading_bot",
    "get_order_details": "trading_bot",
    "get_stock_info": "trading_bot",
    "get_symbol_by_name": "trading_bot",
    "get_transaction_history": "trading_bot",
    "make_transaction": "trading_bot",
    "notify_price_change": "trading_bot",
    "place_order": "trading_bot",
    "remove_stock_from_watchlist": "trading_bot",
    "trading_login": "trading_bot",
    "update_market_status": "trading_bot",
    "update_stock_price": "trading_bot",
    "authenticate_travel": "travel_booking",
    "book_flight": "travel_booking",
    "cancel_booking": "travel_booking",
    "compute_exchange_rate": "travel_booking",
    "contact_customer_support": "travel_booking",
    "get_budget_fiscal_year": "travel_booking",
    "get_credit_card_balance": "travel_booking",
    "get_flight_cost": "travel_booking",
    "get_nearest_airport_by_city": "travel_booking",
    "purchase_insurance": "travel_booking",
    "register_credit_card": "travel_booking",
    "retrieve_invoice": "travel_booking",
    "set_budget_limit": "travel_booking",
    "verify_traveler_information": "travel_booking",
    "activateParkingBrake": "vehicle_control",
    "adjustClimateControl": "vehicle_control",
    "displayCarStatus": "vehicle_control",
    "display_log": "vehicle_control",
    "estimate_distance": "vehicle_control",
    "estimate_drive_feasibility_by_mileage": "vehicle_control",
    "fillFuelTank": "vehicle_control",
    "gallon_to_liter": "vehicle_control",
    "get_zipcode_based_on_city": "vehicle_control",
    "liter_to_gallon": "vehicle_control",
    "lockDoors": "vehicle_control",
    "pressBrakePedal": "vehicle_control",
    "setCruiseControl": "vehicle_control",
    "setHeadlights": "vehicle_control",
    "set_navigation": "vehicle_control",
    "startEngine": "vehicle_control",
}

_ORIGINAL_FILESYSTEM_CONFIG: Config = {
    "home": {
        "type": "directory",
        "contents": {
            "alice": {
                "type": "directory",
                "contents": {
                    "documents": {
                        "type": "directory",
                        "contents": {
                            "readme.txt": {
                                "type": "file",
                                "content": "Welcome to the workspace.\nThis is a shared document area.\nPlease follow the naming conventions.",
                            },
                            "report.csv": {
                                "type": "file",
                                "content": "name,score,grade\nAlice,92,A\nBob,78,B\nCharlie,85,B",
                            },
                        },
                    },
                    "project": {
                        "type": "directory",
                        "contents": {
                            "src": {
                                "type": "directory",
                                "contents": {
                                    "main.py": {
                                        "type": "file",
                                        "content": "def main():\n    print('Hello World')\n\nif __name__ == '__main__':\n    main()",
                                    }
                                },
                            },
                            "config": {
                                "type": "directory",
                                "contents": {
                                    "database.conf": {
                                        "type": "file",
                                        "content": "[database]\nhost=localhost\nport=5432\ndatabase=myapp",
                                    }
                                },
                            },
                            "reports": {
                                "type": "directory",
                                "contents": {
                                    "weekly_summary.pdf": {
                                        "type": "file",
                                        "content": "Weekly Summary Report\nWeek 24\nGenerated on 2024-01-15",
                                    }
                                },
                            },
                        },
                    },
                    "logs": {
                        "type": "directory",
                        "contents": {
                            "activity.log": {
                                "type": "file",
                                "content": "2024-01-15 09:00: Starting process\n2024-01-15 09:30: Processing data\n2024-01-15 10:00: Completed successfully",
                            },
                            "system.log": {
                                "type": "file",
                                "content": "Jan 15 09:00: server sshd[123]: Started\nJan 15 09:15: server kernel: eth0: link up\nJan 15 09:30: server app[456]: Connection timeout\nJan 15 09:45: server app[456]: Retrying connection",
                            },
                        },
                    },
                    "Archive": {"type": "directory", "contents": {}},
                    "notes.txt": {
                        "type": "file",
                        "content": "Meeting notes from 2024-01-15\nAction items: review PR, update docs",
                    },
                },
            }
        },
    },
    "tmp": {
        "type": "directory",
        "contents": {
            "temp_log.txt": {
                "type": "file",
                "content": "Process started at 09:00\nProcess completed at 17:30\nNo errors detected.",
            }
        },
    },
    "documents": {
        "type": "directory",
        "contents": {
            "readme.txt": {
                "type": "file",
                "content": "Welcome to the workspace.\nThis is a shared document area.\nPlease follow the naming conventions.",
            },
            "report.csv": {
                "type": "file",
                "content": "name,score,grade\nAlice,92,A\nBob,78,B\nCharlie,85,B",
            },
        },
    },
}


def build_full_initial_configs(message_config: Config) -> ConfigMap:
    """Build the legacy full initial-state mapping without sharing pool objects."""
    return {
        "GorillaFileSystem": copy.deepcopy(_ORIGINAL_FILESYSTEM_CONFIG),
        "MathAPI": {"numbers": [275.5, 299.75, 250.65, 310.85, 290.1]},
        "MessageAPI": copy.deepcopy(message_config),
        "PostingAPI": copy.deepcopy(POSTING_CONFIGS[0]),
        "TicketAPI": copy.deepcopy(TICKET_CONFIGS[0]),
        "TradingBot": copy.deepcopy(TRADING_CONFIGS[0]),
        "TravelAPI": copy.deepcopy(TRAVEL_CONFIGS[0]),
        "VehicleControlAPI": copy.deepcopy(VEHICLE_CONFIGS[0]),
    }


def get_relevant_class_keys(
    tool_names: Sequence[str], tool_name_to_class_key: Mapping[str, str]
) -> set[str]:
    """Return class keys whose APIs occur in ``tool_names``."""
    return {
        tool_name_to_class_key[name]
        for name in tool_names
        if name in tool_name_to_class_key
    }


def get_canonical_initial_configs(examples: list[ToolSchema]) -> ConfigMap:
    """Extract the largest initial config observed for each class."""
    configs_by_class: defaultdict[str, list[Config]] = defaultdict(list)
    for example in examples:
        initial_config = example.get("initial_config")
        if not is_string_object_dict(initial_config):
            continue
        for class_name, config in initial_config.items():
            if is_string_object_dict(config):
                configs_by_class[class_name].append(config)
    return {
        class_name: max(configs, key=lambda config: len(json.dumps(config)))
        for class_name, configs in configs_by_class.items()
    }


def build_api_name_to_class_key_map(
    tools_data: list[ToolSchema],
) -> dict[str, str]:
    """Build an API-name to class-key mapping from BFCL definitions."""
    mapping: dict[str, str] = {}
    for tool in tools_data:
        api_name = string_field(tool, "api_name")
        class_key = string_field(tool, "tool_name")
        if api_name and class_key:
            mapping[api_name] = class_key
    return mapping
