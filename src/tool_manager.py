from function_schema import get_function_schema
from typing import Any, Dict, List, Optional, Tuple
import json
import datetime
import os
import copy
import inspect
import importlib
from pathlib import Path
from collections import defaultdict

from llm_client import LLMClient
from config_pool import generate_random_config, MESSAGE_CONFIGS
import random


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

FULL_INITIAL_CONFIGS = {
    "GorillaFileSystem": {
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
                                    "content": "Welcome to the workspace.\nThis is a shared document area.\nPlease follow the naming conventions."
                                },
                                "report.csv": {
                                    "type": "file",
                                    "content": "name,score,grade\nAlice,92,A\nBob,78,B\nCharlie,85,B"
                                }
                            }
                        },
                        "project": {
                            "type": "directory",
                            "contents": {
                                "src": {
                                    "type": "directory",
                                    "contents": {
                                        "main.py": {
                                            "type": "file",
                                            "content": "def main():\n    print('Hello World')\n\nif __name__ == '__main__':\n    main()"
                                        }
                                    }
                                },
                                "config": {
                                    "type": "directory",
                                    "contents": {
                                        "database.conf": {
                                            "type": "file",
                                            "content": "[database]\nhost=localhost\nport=5432\ndatabase=myapp"
                                        }
                                    }
                                },
                                "reports": {
                                    "type": "directory",
                                    "contents": {
                                        "weekly_summary.pdf": {
                                            "type": "file",
                                            "content": "Weekly Summary Report\nWeek 24\nGenerated on 2024-01-15"
                                        }
                                    }
                                }
                            }
                        },
                        "logs": {
                            "type": "directory",
                            "contents": {
                                "activity.log": {
                                    "type": "file",
                                    "content": "2024-01-15 09:00: Starting process\n2024-01-15 09:30: Processing data\n2024-01-15 10:00: Completed successfully"
                                },
                                "system.log": {
                                    "type": "file",
                                    "content": "Jan 15 09:00: server sshd[123]: Started\nJan 15 09:15: server kernel: eth0: link up\nJan 15 09:30: server app[456]: Connection timeout\nJan 15 09:45: server app[456]: Retrying connection"
                                }
                            }
                        },
                        "Archive": {
                            "type": "directory",
                            "contents": {}
                        },
                        "notes.txt": {
                            "type": "file",
                            "content": "Meeting notes from 2024-01-15\nAction items: review PR, update docs"
                        }
                    }
                }
            }
        },
        "tmp": {
            "type": "directory",
            "contents": {
                "temp_log.txt": {
                    "type": "file",
                    "content": "Process started at 09:00\nProcess completed at 17:30\nNo errors detected."
                }
            }
        },
        "documents": {
            "type": "directory",
            "contents": {
                "readme.txt": {
                    "type": "file",
                    "content": "Welcome to the workspace.\nThis is a shared document area.\nPlease follow the naming conventions."
                },
                "report.csv": {
                    "type": "file",
                    "content": "name,score,grade\nAlice,92,A\nBob,78,B\nCharlie,85,B"
                }
            }
        }
    },
    "MathAPI": {
        "numbers": [275.5, 299.75, 250.65, 310.85, 290.1]
    },
    "MessageAPI": copy.deepcopy(random.choice(MESSAGE_CONFIGS)),
    "PostingAPI": {
        "authenticated": False,
        "tweet_counter": 17,
        "tweets": {
            "0": {
                "id": 0,
                "username": "genealogy_enthusiast",
                "content": "Excited to start my genealogy journey!",
                "tags": ["#genealogy", "#familyhistory", "#beginnings"],
                "mentions": []
            },
            "1": {
                "id": 1,
                "username": "genealogy_enthusiast",
                "content": "Researching family history is so rewarding.",
                "tags": ["#genealogy", "#research", "#familyhistory"],
                "mentions": []
            },
            "2": {
                "id": 2,
                "username": "genealogy_enthusiast",
                "content": "Can't wait to uncover new stories about my ancestors.",
                "tags": ["#ancestors", "#familystories", "#discovery"],
                "mentions": []
            },
            "3": {
                "id": 3,
                "username": "genealogy_enthusiast",
                "content": "Genealogy is like a puzzle waiting to be solved.",
                "tags": ["#genealogy", "#puzzle", "#research"],
                "mentions": []
            },
            "4": {
                "id": 4,
                "username": "genealogy_enthusiast",
                "content": "Every family has a story worth telling.",
                "tags": ["#familystories", "#heritage", "#genealogy"],
                "mentions": []
            },
            "5": {
                "id": 5,
                "username": "genealogy_enthusiast",
                "content": "Exploring my roots is a journey of self-discovery.",
                "tags": ["#roots", "#selfdiscovery", "#familyhistory"],
                "mentions": []
            },
            "6": {
                "id": 6,
                "username": "genealogy_enthusiast",
                "content": "Family history is a treasure trove of stories.",
                "tags": ["#familyhistory", "#stories", "#heritage"],
                "mentions": []
            },
            "7": {
                "id": 7,
                "username": "genealogy_enthusiast",
                "content": "Connecting with my past to understand my present.",
                "tags": ["#connection", "#pastpresent", "#genealogy"],
                "mentions": []
            },
            "8": {
                "id": 8,
                "username": "genealogy_enthusiast",
                "content": "Genealogy: where history meets personal stories.",
                "tags": ["#genealogy", "#history", "#personalstories"],
                "mentions": []
            },
            "9": {
                "id": 9,
                "username": "genealogy_enthusiast",
                "content": "Uncovering the past, one ancestor at a time.",
                "tags": ["#ancestors", "#research", "#familyhistory"],
                "mentions": []
            },
            "10": {
                "id": 10,
                "username": "tech_user",
                "content": "Artificial intelligence is transforming how we work and live.",
                "tags": ["#AI", "#technology", "#future"],
                "mentions": []
            },
            "11": {
                "id": 11,
                "username": "tech_user",
                "content": "Just published a new blog post about machine learning trends.",
                "tags": ["#machinelearning", "#AI", "#blog"],
                "mentions": []
            },
            "12": {
                "id": 12,
                "username": "tech_user",
                "content": "Exploring the latest breakthroughs in natural language processing.",
                "tags": ["#NLP", "#AI", "#research"],
                "mentions": []
            },
            "13": {
                "id": 13,
                "username": "tech_user",
                "content": "The future of artificial intelligence is bright and full of possibilities.",
                "tags": ["#AI", "#future", "#technology"],
                "mentions": []
            },
            "14": {
                "id": 14,
                "username": "history_buff",
                "content": "Ancient civilizations had remarkable engineering skills.",
                "tags": ["#history", "#engineering", "#ancient"],
                "mentions": []
            },
            "15": {
                "id": 15,
                "username": "family_finder",
                "content": "Found a new branch of my family tree today!",
                "tags": ["#familytree", "#genealogy", "#discovery"],
                "mentions": []
            },
            "16": {
                "id": 16,
                "username": "puzzle_solver",
                "content": "AI-powered tools make data analysis so much easier.",
                "tags": ["#AI", "#data", "#analytics"],
                "mentions": []
            }
        },
        "comments": {
            "0": [
                {"username": "history_buff", "content": "Great start!"},
                {"username": "family_finder", "content": "Welcome to the community!"}
            ],
            "3": [
                {"username": "puzzle_solver", "content": "So true!"}
            ]
        },
        "retweets": [
            {"username": "history_buff", "tweet_id": 0},
            {"username": "family_finder", "tweet_id": 5}
        ],
        "following_list": ["history_buff", "family_finder", "puzzle_solver"],
        "users": {
            "history_buff": {"tweet_count": 25, "following_count": 50, "retweet_count": 10},
            "family_finder": {"tweet_count": 15, "following_count": 30, "retweet_count": 5},
            "puzzle_solver": {"tweet_count": 8, "following_count": 12, "retweet_count": 2}
        },
        "username": "tech_user",
        "password": "testpass"
    },
    "TicketAPI": {
        "tickets_queue": [
            {
                "id": 123456,
                "title": "System Error",
                "description": "There is a critical system error that needs immediate attention.",
                "status": "Open",
                "priority": 4,
                "created_by": "support_agent"
            },
            {
                "id": 654321,
                "title": "Feature Request",
                "description": "Request for a new feature in the application.",
                "status": "In Progress",
                "priority": 2,
                "created_by": "support_agent"
            }
        ],
        "ticket_count": 654321,
        "ticket_counter": 654321,
        "current_user": "",
        "authenticated": False,
        "username": "support_agent",
        "password": "testpass"
    },
    "TradingBot": {
        "account_info": {
            "account_id": 12345,
            "balance": 10000.0,
            "binding_card": 1974202140965533
        },
        "authenticated": False,
        "market_status": "Open",
        "order_counter": 12446,
        "username": "trader_admin",
        "password": "testpass",
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
            "ZETA": {"price": 150.45, "percent_change": 0.05, "volume": 1.789, "MA(5)": 150.0, "MA(20)": 149.5}
        },
        "watch_list": ["NVDA", "ZETA"],
        "transaction_history": [
            {
                "order_id": 12346,
                "symbol": "GOOG",
                "price": 2840.34,
                "num_shares": 5,
                "status": "Pending",
                "timestamp": "2024-10-27 14:10:53"
            }
        ]
    },
    "TravelAPI": {
        "credit_card_list": {
            "12345": {
                "card_number": "123456",
                "expiration_date": "12/2028",
                "cardholder_name": "Michael Smith",
                "card_verification_number": 465,
                "balance": 50000.0
            }
        },
        "booking_record": {
            "flight_001": {
                "travel_to": "Rome",
                "travel_from": "New York",
                "insurance": "none",
                "travel_cost": 1200.5,
                "travel_date": "2024-08-08",
                "travel_class": "Business",
                "transaction_id": "12345",
                "card_id": "12345"
            }
        },
        "access_token": "",
        "token_type": "Bearer",
        "token_expires_in": 0,
        "token_scope": "",
        "user_first_name": "Michael",
        "user_last_name": "Smith",
        "budget_limit": 5000.0,
        "client_id": "travel_client_001",
        "client_secret": "test_secret",
        "refresh_token": "refresh_abc123"
    },
    "VehicleControlAPI": {
        "remainingUnlockedDoors": 0,
        "fuelLevel": 15.0,
        "batteryVoltage": 12.8,
        "engineState": "stopped",
        "doorStatus": {
            "driver": "locked",
            "passenger": "locked",
            "rear_left": "locked",
            "rear_right": "locked"
        },
        "acTemperature": 22.0,
        "fanSpeed": 60,
        "acMode": "auto",
        "humidityLevel": 45.0,
        "headLightStatus": "off",
        "parkingBrakeStatus": "released",
        "parkingBrakeForce": 0.0,
        "slopeAngle": 0.0,
        "distanceToNextVehicle": 100.0,
        "cruiseStatus": "inactive",
        "destination": "Grand Canyon",
        "frontLeftTirePressure": 35.0,
        "frontRightTirePressure": 35.0,
        "rearLeftTirePressure": 35.0,
        "rearRightTirePressure": 35.0
    }
}

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


def get_relevant_class_keys(tool_names: List[str]) -> set:
    """Return the set of class_keys whose APIs are used by the given tool names."""
    keys = set()
    for t in tool_names:
        if t in TOOL_NAME_TO_CLASS_KEY:
            keys.add(TOOL_NAME_TO_CLASS_KEY[t])
    return keys


def filter_api_state(
    full_state: Dict[str, Dict[str, Any]],
    tool_names: List[str],
) -> Dict[str, Dict[str, Any]]:
    """Return only the class_key entries relevant to *tool_names*.

    If *tool_names* is empty, returns the full state unchanged (safe default).
    """
    relevant = get_relevant_class_keys(tool_names)
    if not relevant:
        return full_state
    return {k: v for k, v in full_state.items() if k in relevant}


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


def get_canonical_initial_configs(examples: List[Dict]) -> Dict[str, Dict[str, Any]]:
    """Extract the most representative (largest) initial_config per class."""
    configs_by_class = defaultdict(list)
    for ex in examples:
        ic = ex.get("initial_config", {})
        for cls_name, cls_config in ic.items():
            if isinstance(cls_config, dict):
                configs_by_class[cls_name].append(cls_config)
    canonical = {}
    for cls_name, configs in configs_by_class.items():
        largest = max(configs, key=lambda x: len(json.dumps(x)))
        canonical[cls_name] = largest
    return canonical


def create_python_tool_instances(
    canonical_configs: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """Instantiate all 8 Python tool classes with canonical initial configs.

    Returns:
        Dict mapping class_key -> tool instance
    """
    instances = {}
    for class_key in TOOL_CLASS_KEYS:
        config_key = CLASS_KEY_TO_INITIAL_CONFIG_KEY[class_key]
        config = copy.deepcopy(canonical_configs.get(config_key, {}))
        try:
            mod = importlib.import_module(f"tools.{class_key}")
            cls_name = CLASS_KEY_TO_CLASS_NAME[class_key]
            cls = getattr(mod, cls_name)
            instances[class_key] = cls(initial_config=config)
        except Exception as e:
            print(f"Warning: Could not instantiate {class_key}: {e}")
    return instances


def build_api_name_to_class_key_map(tools_data: List[Dict]) -> Dict[str, str]:
    """Build a mapping from api_name -> class_key using the tool definitions.

    Each tool in the JSONL has 'api_name' and 'tool_name' (which is the class_key).
    """
    mapping = {}
    for tool in tools_data:
        api_name = tool.get("api_name", "")
        class_key = tool.get("tool_name", "")
        if api_name and class_key:
            mapping[api_name] = class_key
    return mapping


# >>>>> Example functions (kept for reference) <<<<<

def create_calendar_event(
    summary: str,
    start_time: str,
    end_time: str,
) -> None:
    """Creates a new calendar event."""
    pass


def fetch_calendar_events(
    start_date: str,
    end_date: str,
) -> str:
    """
    Retrieves calendar events within a specified date range.
    """
    pass


def web_search(query: str) -> str:
    """
    Searches the web (DuckDuckGo) for the given query.
    Returns a JSON string containing search results.
    """
    pass


# <<<<< Example functions <<<<<


class ToolManager:
    """
    Manages a pool of tools that can be loaded from a file or defined in code.
    
    Supports loading tools from:
    1. A JSON/JSONL file containing BFCL-style tool definitions
    2. Python functions with type hints and docstrings
    """
    
    def __init__(
        self,
        llm: LLMClient,
        tool_pool_path: Optional[str] = None,
        tools: Optional[List] = None,
        invocation_examples_path: Optional[str] = None,
        use_config_pool: bool = True,
    ):
        """
        Initialize the ToolManager.

        Args:
            llm: LLMClient instance for simulating tool execution
            tool_pool_path: Path to a JSON or JSONL file containing tool definitions.
            If provided, tools will be loaded from this file.
            tools: Optional list of Python functions to use as tools.
            If both tool_pool_path and tools are provided, they will be merged.
            invocation_examples_path: Path to bfcl_v3_invocation_examples.jsonl.
            If provided, Python tool implementations will be loaded and used
            for actual execution instead of LLM simulation.
            use_config_pool: If True, reset_python_tool_instances() picks a random
            config from the config pool for diverse starting states. If False,
            falls back to the single FULL_INITIAL_CONFIGS.
        """
        self.llm = llm
        self.use_config_pool = use_config_pool
        self.tool_schemas: List[Dict[str, Any]] = []
        self.tool_implementations: Dict[str, Any] = {}

        # Python tool instances: class_key -> instance
        self.python_tool_instances: Dict[str, Any] = {}
        # Mapping from api_name -> class_key
        self.api_name_to_class_key: Dict[str, str] = {}
        # Canonical initial configs for resetting instances
        self._canonical_configs: Dict[str, Dict[str, Any]] = {}

        # Load tools from file if path is provided
        if tool_pool_path:
            self._load_tools_from_file(tool_pool_path)

        # Load tools from Python functions if provided
        if tools:
            self._load_tools_from_functions(tools)

        # If neither file nor functions provided, use default example tools
        if not self.tool_schemas:
            self._load_default_tools()

        # Load Python tool implementations if invocation examples provided
        if invocation_examples_path:
            self.load_python_tool_implementations(invocation_examples_path)

    def load_python_tool_implementations(self, invocation_examples_path: str) -> None:
        """Load Python tool implementations from invocation examples.

        This creates instances of all 8 tool classes and builds the
        api_name -> class_key mapping so that invoke_tool can call
        actual Python code instead of LLM simulation.

        Args:
            invocation_examples_path: Path to bfcl_v3_invocation_examples.jsonl
        """
        path_obj = Path(invocation_examples_path)
        if not path_obj.exists():
            print(f"Warning: Invocation examples file not found: {invocation_examples_path}")
            return

        # Load invocation examples
        examples = []
        with open(path_obj, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        examples.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        # Get canonical initial configs
        self._canonical_configs = get_canonical_initial_configs(examples)

        # Build api_name -> class_key mapping from loaded tool schemas
        for tool in self.tool_schemas:
            # The 'name' field in tool_schema is the api_name
            # We need to find the class_key. We can match by looking at
            # the original BFCL data stored in tool_implementations
            pass

        # Build the mapping from the BFCL definitions we already loaded
        self.api_name_to_class_key = {}
        for tool_name, impl_info in self.tool_implementations.items():
            if impl_info.get("type") == "bfcl":
                bfcl_data = impl_info.get("data", {})
                class_key = bfcl_data.get("tool_name", "")
                if class_key:
                    self.api_name_to_class_key[tool_name] = class_key

        # Create Python tool instances
        self.python_tool_instances = create_python_tool_instances(self._canonical_configs)

        loaded = len(self.python_tool_instances)
        mapped = len(self.api_name_to_class_key)
        print(f"Loaded {loaded} Python tool classes with {mapped} api_name mappings")

    def reset_python_tool_instances(self) -> None:
        """Reset all Python tool instances to fresh state.

        If self.use_config_pool is True, picks a random config from the pool
        for each domain, giving diverse starting states across datapoints.
        Otherwise falls back to the original single FULL_INITIAL_CONFIGS.
        
        If _cached_initial_config is set, reuses that config instead of
        generating a new random one. This ensures consistency across retries
        within a single datapoint generation.
        """
        if hasattr(self, '_cached_initial_config') and self._cached_initial_config is not None:
            config = self._cached_initial_config
        elif self.use_config_pool:
            config = generate_random_config()
        else:
            config = FULL_INITIAL_CONFIGS
        self.python_tool_instances = create_python_tool_instances(config)

    def initialize_api_state(self, force_new: bool = False) -> None:
        """Initialize all Python tool instances with full, realistic API state.

        This is the primary method to call at the beginning of each datapoint
        generation. It ensures all stateful APIs (Message, Posting, Ticket,
        Trading, Travel) start with a complete, consistent state so that
        login calls and subsequent operations succeed properly.

        For example, MessageAPI gets pre-existing users in user_map so that
        message_login with a valid user ID succeeds, and PostingAPI gets
        username/password credentials so authenticate_twitter works.
        
        Args:
            force_new: If True, forces selection of a new random config even
                if a config is cached. Use this when starting a fresh datapoint.
                If False (default), reuses cached config for consistency.
        """
        if force_new or not hasattr(self, '_cached_initial_config') or self._cached_initial_config is None:
            # Pick and cache a new random config
            if self.use_config_pool:
                self._cached_initial_config = generate_random_config()
            else:
                self._cached_initial_config = FULL_INITIAL_CONFIGS
        self.reset_python_tool_instances()

    def clear_cached_config(self) -> None:
        """Clear the cached initial config to force a new random config on next initialize_api_state.
        
        Call this at the start of each new datapoint generation to ensure
        diversity across datapoints while maintaining consistency within
        a single datapoint (including retries).
        """
        if hasattr(self, '_cached_initial_config'):
            self._cached_initial_config = None

    def get_api_state(self) -> Dict[str, Dict[str, Any]]:
        """Snapshot the current state of all Python tool instances.

        Returns a dict mapping class_key -> state_dict for every instantiated
        tool class.  State dicts are JSON-serialisable (nested structures of
        primitives, lists, dicts).  Non-serialisable values are converted to
        strings via ``default=str``.

        The snapshot is a deep copy so callers can freely mutate it without
        affecting the live instances.
        """
        import json as _json

        state: Dict[str, Dict[str, Any]] = {}
        for class_key, instance in self.python_tool_instances.items():
            raw = vars(instance)
            # Round-trip through JSON to guarantee serialisability & deep copy
            try:
                state[class_key] = _json.loads(_json.dumps(raw, default=str))
            except (TypeError, ValueError):
                # Fallback: str-ify anything that fails
                state[class_key] = _json.loads(_json.dumps(
                    {k: str(v) for k, v in raw.items()}
                ))
        return state

    def restore_api_state(self, state: Dict[str, Dict[str, Any]]) -> None:
        """Restore API state from a snapshot.

        This method takes a state dict (as returned by get_api_state) and
        restores it to the live Python tool instances. This is used for
        checkpoint/resume functionality to restore state after loading
        a partial generation.

        Args:
            state: Dict mapping class_key -> state_dict
        """
        if not state:
            return

        for class_key, instance_state in state.items():
            if class_key not in self.python_tool_instances:
                continue

            instance = self.python_tool_instances[class_key]
            for key, value in instance_state.items():
                try:
                    setattr(instance, key, value)
                except (AttributeError, TypeError) as e:
                    print(f"  Warning: Could not restore {class_key}.{key}: {e}")

    def has_python_implementation(self, tool_name: str) -> bool:
        """Check if a tool has a Python implementation available.

        Args:
            tool_name: The api_name of the tool

        Returns:
            True if a Python implementation is available
        """
        class_key = self.api_name_to_class_key.get(tool_name)
        return class_key is not None and class_key in self.python_tool_instances

    def invoke_python_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """Invoke a Python tool implementation directly.

        Args:
            tool_name: The api_name of the tool
            params: Parameters to pass to the tool method

        Returns:
            The result of the Python tool invocation

        Raises:
            ValueError: If no Python implementation exists for the tool
        """
        class_key = self.api_name_to_class_key.get(tool_name)
        if class_key is None or class_key not in self.python_tool_instances:
            raise ValueError(f"No Python implementation for tool '{tool_name}'")

        instance = self.python_tool_instances[class_key]
        method = getattr(instance, tool_name, None)
        if method is None or not callable(method):
            raise ValueError(f"Method '{tool_name}' not found on {class_key}")

        try:
            coerced_params = self._coerce_params(method, params)
            return method(**coerced_params)
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def _coerce_params(method: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        """Coerce parameter types to match the method signature.

        LLM-generated arguments may have incorrect types (e.g., strings
        where ints are expected). This method inspects the function
        signature and attempts type conversion.

        Args:
            method: The callable method
            params: The parameters to coerce

        Returns:
            Parameters with types coerced to match the method signature
        """
        sig = inspect.signature(method)
        coerced = {}
        for key, value in params.items():
            if key not in sig.parameters:
                coerced[key] = value
                continue

            param = sig.parameters[key]
            annotation = param.annotation

            if annotation is inspect.Parameter.empty:
                coerced[key] = value
                continue

            # Try to coerce the value to the annotated type
            try:
                if annotation is int and not isinstance(value, int):
                    coerced[key] = int(value)
                elif annotation is float and not isinstance(value, float):
                    coerced[key] = float(value)
                elif annotation is bool and not isinstance(value, bool):
                    if isinstance(value, str):
                        coerced[key] = value.lower() in ('true', '1', 'yes')
                    else:
                        coerced[key] = bool(value)
                elif annotation is list and not isinstance(value, list):
                    if isinstance(value, str):
                        coerced[key] = json.loads(value)
                    else:
                        coerced[key] = list(value) if value else []
                elif annotation is dict and not isinstance(value, dict):
                    if isinstance(value, str):
                        coerced[key] = json.loads(value)
                    else:
                        coerced[key] = value
                else:
                    coerced[key] = value
            except (ValueError, TypeError, json.JSONDecodeError):
                coerced[key] = value

        return coerced
    
    def _load_tools_from_file(self, path: str) -> None:
        """
        Load tools from a JSON or JSONL file.
        
        Args:
            path: Path to the tool definition file
        """
        path_obj = Path(path)
        
        if not path_obj.exists():
            raise FileNotFoundError(f"Tool pool file not found: {path}")
        
        with open(path_obj, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        tools_data = []
        
        # Try to parse as JSON first, then as JSONL
        try:
            # Try parsing as JSON array
            tools_data = json.loads(content)
            if not isinstance(tools_data, list):
                tools_data = [tools_data]
        except json.JSONDecodeError:
            # Parse as JSONL (one JSON object per line)
            for line in content.split('\n'):
                line = line.strip()
                if line:
                    try:
                        tools_data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping invalid JSON line: {line[:50]}... Error: {e}")
        
        for tool_data in tools_data:
            self._add_tool_from_bfcl_definition(tool_data)
    
    def _add_tool_from_bfcl_definition(self, tool_data: Dict[str, Any]) -> None:
        """
        Add a tool from BFCL-style definition.
        
        Args:
            tool_data: Dictionary containing tool definition
        """
        tool_name = tool_data.get('api_name') or tool_data.get('name')
        if not tool_name:
            print(f"Warning: Skipping tool without name: {tool_data}")
            return
        
        # Build the schema in the format expected by the system
        parameters = tool_data.get('parameters', {})
        properties = parameters.get('properties', {})
        required = parameters.get('required', [])
        optional = parameters.get('optional', [])
        
        # Build parameter schema
        param_schema = {
            "type": "object",
            "properties": {},
            "required": required
        }
        
        # Convert properties
        for param_name, param_info in properties.items():
            if isinstance(param_info, dict):
                param_type = param_info.get('type', 'string')
                # Map types to JSON Schema types
                type_mapping = {
                    'STRING': 'string',
                    'NUMBER': 'number',
                    'INTEGER': 'integer',
                    'FLOAT': 'number',
                    'BOOLEAN': 'boolean',
                    'ARRAY': 'array',
                    'OBJECT': 'object',
                    'DATE': 'string',
                    'TUPLE': 'array',
                }
                json_type = type_mapping.get(param_type.upper(), 'string')
                
                param_schema["properties"][param_name] = {
                    "type": json_type,
                    "description": param_info.get('description', '')
                }
                
                # Add default if present
                if 'default' in param_info:
                    param_schema["properties"][param_name]["default"] = param_info['default']
            else:
                # Simple type definition
                param_schema["properties"][param_name] = {
                    "type": "string",
                    "description": ""
                }
        
        # Create the tool schema with output information
        schema = {
            "name": tool_name,
            "description": tool_data.get('api_description') or tool_data.get('tool_description') or '',
            "parameters": param_schema,
            # Include output type and description for new format
            "output_type": tool_data.get('output_type', 'unknown'),
            "output_description": tool_data.get('output_description', ''),
            "output_schema": tool_data.get('output_schema', {}),
            # Include category for grouping
            "category": tool_data.get('category', 'Unknown')
        }
        
        self.tool_schemas.append(schema)
        
        # Store implementation info for virtual execution
        self.tool_implementations[tool_name] = {
            "type": "bfcl",
            "data": tool_data
        }
    
    def _load_tools_from_functions(self, tools: List) -> None:
        """
        Load tools from a list of Python functions.
        
        Args:
            tools: List of Python function objects
        """
        for tool_func in tools:
            schema = get_function_schema(tool_func)
            self.tool_schemas.append(schema)
            self.tool_implementations[tool_func.__name__] = {
                "type": "python",
                "func": tool_func
            }
    
    def _load_default_tools(self) -> None:
        """Load default example tools."""
        default_tools = [
            create_calendar_event,
            fetch_calendar_events,
            web_search,
        ]
        for tool_func in default_tools:
            schema = get_function_schema(tool_func)
            self.tool_schemas.append(schema)
            self.tool_implementations[tool_func.__name__] = {
                "type": "python",
                "func": tool_func
            }
    
    def get_categories(self) -> List[str]:
        """
        Get a list of unique categories across all tools.

        Returns:
            List[str]: List of unique category names
        """
        categories = set()
        for tool in self.tool_schemas:
            category = tool.get('category', 'Unknown')
            categories.add(category)
        return sorted(list(categories))

    def get_tools_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Get a list of tools that belong to the specified category.

        Args:
            category: The category to filter tools by

        Returns:
            List[Dict[str, Any]]: List of tool schemas in the specified category
        """
        return [tool for tool in self.tool_schemas if tool.get('category') == category]

    def get_tool_category(self, tool_name: str) -> Optional[str]:
        """
        Get the category for a specific tool.

        Args:
            tool_name: The name of the tool

        Returns:
            Optional[str]: The category of the tool, or None if not found
        """
        for tool in self.tool_schemas:
            if tool.get('name') == tool_name:
                return tool.get('category')
        return None

    def get_tools_json_schema(self) -> List[Dict[str, Any]]:
        """Get all tool schemas in JSON format."""
        return self.tool_schemas

    def get_tools_with_descriptions(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get tools with their full descriptions.

        Args:
            category: Optional category to filter by

        Returns:
            List[Dict[str, Any]]: List of tools with descriptions
        """
        if category:
            return self.get_tools_by_category(category)
        return self.tool_schemas

    def get_tool_schema(self, tool_name: str) -> Dict[str, Any]:
        """
        Get the schema for a specific tool by name.

        Args:
            tool_name: The name of the tool to get the schema for

        Returns:
            dict: The schema for the tool
            
        Raises:
            ValueError: If the tool does not exist
        """
        for tool in self.tool_schemas:
            if tool["name"] == tool_name:
                return tool

        available_tools = [tool["name"] for tool in self.tool_schemas]
        raise ValueError(
            f"Tool '{tool_name}' not found. Available tools: {', '.join(available_tools)}"
        )
    
    def tool_exists(self, tool_name: str) -> bool:
        """
        Check if a tool with the given name exists in the available tools.
        
        Args:
            tool_name: The name of the tool to check
            
        Returns:
            bool: True if the tool exists, False otherwise
        """
        return any(tool["name"] == tool_name for tool in self.tool_schemas)
    
    def invoke_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """
        Invoke a tool with the given parameters.

        If a Python implementation is available, it will be used for
        actual execution. Otherwise, falls back to LLM-based simulation.

        Args:
            tool_name: Name of the tool to invoke
            params: Parameters to pass to the tool

        Returns:
            The result of the tool invocation

        Raises:
            ValueError: If the tool does not exist
        """
        # Check if tool exists and get its schema
        tool_schema = None
        for tool in self.tool_schemas:
            if tool["name"] == tool_name:
                tool_schema = tool
                break

        if tool_schema is None:
            available_tools = [tool["name"] for tool in self.tool_schemas]
            raise ValueError(
                f"Tool '{tool_name}' not found. Available tools: {', '.join(available_tools)}"
            )

        # Try Python implementation first (actual tool execution)
        if self.has_python_implementation(tool_name):
            return self.invoke_python_tool(tool_name, params)

        # Get implementation info
        impl_info = self.tool_implementations.get(tool_name)

        if impl_info and impl_info.get("type") == "python":
            # Call the actual Python function
            func = impl_info.get("func")
            if func:
                try:
                    return func(**params)
                except Exception as e:
                    return {"error": str(e)}

        # Use virtual tool executor for BFCL-style tools or when no implementation exists
        return self.__virtual_tool_executor(tool_name, params, schema=tool_schema)
    
    def __virtual_tool_executor(
        self, tool_name: str, params: dict, schema: dict
    ) -> Any:
        """
        Simulate tool execution using LLM with enhanced prompts and retry logic.

        Args:
            tool_name: Name of the tool
            params: Parameters for the tool
            schema: Tool schema

        Returns:
            Simulated tool output
        """
        # Extract output type and description from schema
        output_type = schema.get('output_type', 'unknown')
        output_description = schema.get('output_description', '')

        # Build enhanced output guidance with type-specific examples
        output_guidance = self._build_output_guidance(output_type, output_description)

        # Build the enhanced prompt with few-shot examples
        prompt = self._build_simulation_prompt(
            tool_name=tool_name,
            params=params,
            schema=schema,
            output_guidance=output_guidance
        )

        # Try with retries for validation failures
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                response, _ = self.llm.json_output(
                    prompt=prompt,
                    reasoning=True,
                )

                # Validate the response
                validated_response = self._validate_tool_output(
                    tool_name, response, output_type, output_description
                )

                # Check if validation passed
                if self._is_output_valid(validated_response, output_type, output_description):
                    return validated_response

                if attempt < max_retries:
                    print(f"    Tool simulation validation failed for {tool_name}, retrying ({attempt + 1}/{max_retries})...")
                    # Add correction guidance for retry
                    prompt += f"\n\n=== CORRECTION NEEDED ===\nYour previous output did not match the expected type '{output_type}'.\nPlease regenerate ensuring the output is of type {output_type} and matches the description.\n"

            except Exception as e:
                print(f"    Error simulating tool {tool_name}: {e}")
                if attempt < max_retries:
                    continue
                # Return a sensible default on final failure
                return self._get_default_output(output_type)

        return validated_response if 'validated_response' in locals() else self._get_default_output(output_type)

    def _validate_tool_output(
        self, 
        tool_name: str, 
        output: Any, 
        expected_type: str, 
        output_description: str
    ) -> Any:
        """
        Validate that the simulated tool output matches the declared output type and description.
        
        Args:
            tool_name: Name of the tool
            output: The simulated output to validate
            expected_type: The declared output type
            output_description: The declared output description
            
        Returns:
            The validated output (unchanged if valid)
        """
        if expected_type == 'unknown' or not expected_type:
            # No validation if type is unknown
            return output
            
        # Basic type checking
        type_mapping = {
            'string': str,
            'integer': int,
            'float': (int, float),
            'boolean': bool,
            'list': list,
            'dict': dict,
            'number': (int, float),
        }
        
        # Handle compound types like "integer or string"
        base_type = expected_type.split()[0].lower()  # Take first word
        expected_python_type = type_mapping.get(base_type, None)
        
        if expected_python_type:
            if not isinstance(output, expected_python_type):
                # Try to convert if possible
                try:
                    if base_type in ('integer', 'float', 'number'):
                        if isinstance(output, str):
                            output = float(output) if '.' in output else int(output)
                    elif base_type == 'string':
                        output = str(output)
                    elif base_type == 'boolean':
                        if isinstance(output, str):
                            output = output.lower() in ('true', 'yes', '1')
                    elif base_type == 'list':
                        if isinstance(output, dict):
                            output = list(output.values())
                    elif base_type == 'dict':
                        if isinstance(output, str):
                            output = json.loads(output)
                except (ValueError, TypeError, json.JSONDecodeError):
                    # If conversion fails, return as-is but log warning
                    print(f"Warning: Output type mismatch for {tool_name}. Expected {expected_type}, got {type(output).__name__}")
        
        # Use LLM to verify output matches description
        if output_description and output_description != 'Failed to predict output description':
            validation_prompt = f"""You are an output validator. Given the following tool output and its expected description, determine if the output is plausible and matches the description.

Tool Name: {tool_name}
Expected Output Description: {output_description}
Actual Output: {json.dumps(output, default=str) if isinstance(output, (dict, list)) else str(output)}

Does the output plausibly match the description? Answer YES or NO, followed by a brief explanation.

Response format:
{{
    "VALID": "YES" or "NO",
    "REASON": "<brief explanation>"
}}"""

            try:
                validation_response, _ = self.llm.json_output(
                    prompt=validation_prompt,
                    reasoning=False,
                )
                
                if validation_response:
                    is_valid = validation_response.get('VALID', 'YES').upper() == 'YES'
                    if not is_valid:
                        reason = validation_response.get('REASON', 'No reason provided')
                        print(f"Warning: LLM validation flagged output for {tool_name} as not matching description: {reason}")
                        # Return error to trigger retry instead of returning invalid output
                        return {"error": f"Output validation failed: {reason}", "error_type": "validation_failure"}
            except Exception as e:
                # If validation fails, just log and continue
                print(f"Warning: Could not validate output for {tool_name}: {e}")

        return output

    def _build_output_guidance(self, output_type: str, output_description: str) -> str:
        """Build enhanced output guidance with type-specific examples and field requirements."""
        guidance = ""

        if output_type and output_type != 'unknown':
            guidance += f"\n\n=== REQUIRED OUTPUT TYPE ==="
            guidance += f"\nType: {output_type}"
            guidance += f"\nCRITICAL: You MUST return output of exactly this type. No exceptions."

            # Add type-specific examples
            type_examples = {
                'dict': '{"key": "value", "id": 123, "name": "example"}',
                'list': '[{"item": 1}, {"item": 2}, {"item": 3}]',
                'string': '"your string value here"',
                'integer': '42',
                'float': '3.14',
                'number': '42 or 3.14',
                'boolean': 'true or false',
            }

            base_type = output_type.split()[0].lower()
            if base_type in type_examples:
                guidance += f"\nExample of correct {output_type} output: {type_examples[base_type]}"

        if output_description and output_description != 'Failed to predict output description':
            guidance += f"\n\n=== OUTPUT DESCRIPTION ==="
            guidance += f"\n{output_description}"
            guidance += f"\n"
            guidance += f"\nYOUR OUTPUT MUST SATISFY THIS DESCRIPTION:"
            guidance += f"\n- Study the description carefully and include ALL fields/values mentioned"
            guidance += f"\n- The output content must realistically match what the description promises"
            guidance += f"\n- For dict outputs, ensure all keys mentioned in the description are present"
            guidance += f"\n- Do NOT invent fields that aren't mentioned in the description"

            # Parse description for common field patterns and add explicit requirements
            desc_lower = output_description.lower()
            if 'message_id' in desc_lower or 'message id' in desc_lower:
                guidance += f"\n- MUST include 'message_id' field"
            if 'success' in desc_lower:
                guidance += f"\n- MUST include 'success' field (boolean)"
            if 'timestamp' in desc_lower:
                guidance += f"\n- MUST include 'timestamp' field with ISO format"
            if 'status' in desc_lower:
                guidance += f"\n- MUST include 'status' field"
            if 'id' in desc_lower and base_type == 'dict':
                guidance += f"\n- MUST include an 'id' or identifier field"

        return guidance

    def _get_output_format_instructions(self, output_type: str, output_description: str = "") -> str:
        """Get specific formatting instructions based on the output type and description."""
        base_type = output_type.split()[0].lower() if output_type else 'unknown'

        # Common instructions about matching description
        desc_check = ""
        if output_description and output_description != 'Failed to predict output description':
            desc_check = f"""

MANDATORY OUTPUT CONTENT CHECK:
- Your output MUST match this description: {output_description}
- Include ALL fields mentioned in the description
- Values must be realistic and appropriate for the described output"""

        if base_type == 'dict':
            return f"""REQUIREMENTS:
1. CRITICAL: Return ONLY a JSON OBJECT (dictionary) - NOT a string, NOT a list, NOT a number
2. The JSON object MUST have key-value pairs: {{"field1": "value1", "field2": "value2"}}
3. Example: {{"user_id": "U12345", "name": "John", "status": "active"}}
4. NO markdown formatting, NO code blocks (no ```), NO explanations - output ONLY the raw JSON object
5. For error cases ONLY, return: {{"error": "description", "error_description": "details"}}{desc_check}"""
        elif base_type == 'list':
            return f"""REQUIREMENTS:
1. CRITICAL: Return ONLY a JSON ARRAY (list) - NOT a dict, NOT a string, NOT a number
2. The JSON array MUST be wrapped in square brackets: [item1, item2, item3]
3. Example: [{{"id": 1}}, {{"id": 2}}] or ["item1", "item2"]
4. NO markdown formatting, NO code blocks (no ```), NO explanations - output ONLY the raw JSON array
5. For error cases ONLY, return: {{"error": "description", "error_description": "details"}}{desc_check}"""
        elif base_type == 'string':
            return f"""REQUIREMENTS:
1. CRITICAL: Return ONLY a PLAIN STRING value - NOT a JSON object, NOT a JSON array, NOT a number
2. Example: "U12345" or "Operation completed successfully" or just Hello World (without quotes is also acceptable)
3. The output should be the string value itself, optionally wrapped in quotes
4. NO markdown formatting, NO code blocks (no ```), NO explanations - output ONLY the raw string
5. CRITICAL: Do NOT wrap the string in a dict like {{"result": "string"}} - return ONLY the string
6. For error cases ONLY, return a JSON object: {{"error": "description"}}{desc_check}"""
        elif base_type in ('integer', 'float', 'number'):
            return f"""REQUIREMENTS:
1. CRITICAL: Return ONLY a NUMERIC value (integer or float) - NOT a string, NOT a dict, NOT a list
2. Example: 42 or 3.14
3. NO quotes around the number - output the raw number only
4. NO markdown formatting, NO code blocks, NO explanations - output ONLY the raw number
5. For error cases ONLY, return: {{"error": "description", "error_description": "details"}}{desc_check}"""
        elif base_type == 'boolean':
            return f"""REQUIREMENTS:
1. CRITICAL: Return ONLY a BOOLEAN value: true or false (lowercase, no quotes) - NOT a string "true"
2. Example: true (NOT "true", NOT {{"result": true}})
3. NO markdown formatting, NO code blocks, NO explanations - output ONLY the raw boolean
4. For error cases ONLY, return: {{"error": "description", "error_description": "details"}}{desc_check}"""
        else:
            return f"""REQUIREMENTS:
1. CRITICAL: Return a value matching the REQUIRED OUTPUT TYPE specified above
2. NO markdown formatting, NO code blocks, NO explanations - output ONLY the raw value
3. For error cases, return: {{"error": "description", "error_description": "details"}}{desc_check}"""

    def _build_simulation_prompt(self, tool_name: str, params: dict, schema: dict, output_guidance: str) -> str:
        """Build the enhanced simulation prompt with examples."""

        # Build few-shot examples based on tool type
        examples = self._get_few_shot_examples(tool_name, schema)

        # Determine output format instructions based on output type and description
        output_type = schema.get('output_type', 'unknown')
        output_description = schema.get('output_description', '')
        output_format_instructions = self._get_output_format_instructions(output_type, output_description)

        prompt = f"""You are an expert function simulator. Simulate the execution of the following function call.

=== FUNCTION DETAILS ===
Function Name: {tool_name}

Function Description: {schema.get('description', 'No description available')}

Function Parameters Schema:
{json.dumps(schema.get('parameters', {}), indent=2)}
{output_guidance}

=== ARGUMENTS PROVIDED ===
{json.dumps(params, indent=2, default=str)}

Current Date/Time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

{examples}

=== YOUR TASK ===
Generate the return value that '{tool_name}' would produce if executed with the given arguments.

{output_format_instructions}

Generate the output now:"""

        return prompt

    def _get_few_shot_examples(self, tool_name: str, schema: dict) -> str:
        """Generate few-shot examples based on the tool type."""
        output_type = schema.get('output_type', 'unknown')

        # Examples for common tool patterns
        examples = {
            'dict': '''
=== EXAMPLES ===

Example 1 - Tool returning user info (dict):
Function: get_user
Arguments: {"user_id": "U123"}
Expected Type: dict
Expected Description: Returns user information including user_id, username, email, created_at
Output:
{"user_id": "U123", "username": "john_doe", "email": "john@example.com", "created_at": "2024-01-15T10:30:00Z", "status": "active"}

Example 2 - Tool creating a resource (dict):
Function: create_ticket
Arguments: {"title": "Issue with login", "priority": "high"}
Expected Type: dict
Expected Description: Returns created ticket details with ticket_id, status
Output:
{"ticket_id": "TKT-2024-001", "title": "Issue with login", "priority": "high", "status": "open", "created_at": "2024-01-20T14:30:00Z"}''',

            'list': '''
=== EXAMPLES ===

Example 1 - Tool returning list of items:
Function: list_files
Arguments: {"directory": "/home/user"}
Expected Type: list
Expected Description: Returns list of filenames in the directory
Output:
["document.txt", "photo.jpg", "data.csv", "notes.md"]

Example 2 - Tool returning list of objects:
Function: get_tweet_comments
Arguments: {"tweet_id": "12345"}
Expected Type: list
Expected Description: Returns list of comments with user info and text
Output:
[{"comment_id": "C001", "user": "@alice", "text": "Great post!", "timestamp": "2024-01-20T10:00:00Z"}, {"comment_id": "C002", "user": "@bob", "text": "Thanks for sharing", "timestamp": "2024-01-20T11:30:00Z"}]''',

            'string': '''
=== EXAMPLES ===

Example 1 - Tool returning a message:
Function: generate_welcome_message
Arguments: {"username": "Alice"}
Expected Type: string
Expected Description: Returns a personalized welcome message
Output:
"Welcome to our platform, Alice! We're excited to have you join us."''',

            'integer': '''
=== EXAMPLES ===

Example 1 - Tool returning a count:
Function: count_lines
Arguments: {"file": "data.txt"}
Expected Type: integer
Expected Description: Returns the number of lines in the file
Output:
42''',

            'float': '''
=== EXAMPLES ===

Example 1 - Tool returning a price:
Function: calculate_exchange_rate
Arguments: {"from": "USD", "to": "EUR"}
Expected Type: float
Expected Description: Returns the current exchange rate
Output:
0.9234'''
        }

        base_type = output_type.split()[0].lower() if output_type else 'unknown'
        return examples.get(base_type, "")

    def _is_output_valid(self, output: Any, expected_type: str, expected_description: str) -> bool:
        """Quick validation check to see if output matches expected type."""
        if expected_type == 'unknown' or not expected_type:
            return True

        # Basic type checking
        base_type = expected_type.split()[0].lower()

        type_checks = {
            'dict': lambda x: isinstance(x, dict),
            'list': lambda x: isinstance(x, list),
            'string': lambda x: isinstance(x, str),
            'integer': lambda x: isinstance(x, int) and not isinstance(x, bool),
            'float': lambda x: isinstance(x, float),
            'number': lambda x: isinstance(x, (int, float)) and not isinstance(x, bool),
            'boolean': lambda x: isinstance(x, bool),
        }

        check_func = type_checks.get(base_type)
        if check_func:
            return check_func(output)

        return True

    def _get_default_output(self, output_type: str) -> Any:
        """Return a sensible default output for the given type."""
        defaults = {
            'dict': {"status": "success", "message": "Operation completed"},
            'list': [],
            'string': "success",
            'integer': 0,
            'float': 0.0,
            'number': 0,
            'boolean': True,
        }

        base_type = output_type.split()[0].lower() if output_type else 'unknown'
        return defaults.get(base_type, {"status": "completed"})


# Default tools for backward compatibility (can be imported if needed)
DEFAULT_TOOLS = [
    create_calendar_event,
    fetch_calendar_events,
    web_search,
]


if __name__ == "__main__":
    # Example: Load tools from BFCL tool pool file with Python implementations
    project_root = os.path.join(os.path.dirname(__file__), "..", "..")
    tool_pool_path = os.path.join(
        project_root,
        "magnet_tool_extraction",
        "bfcl_v3_tools_with_outputs.jsonl"
    )
    invocation_examples_path = os.path.join(
        project_root,
        "magnet_tool_extraction",
        "bfcl_v3_invocation_examples.jsonl"
    )

    print("Initializing ToolManager with tool pool + Python implementations...")
    tool_manager = ToolManager(
        llm=LLMClient(),
        tool_pool_path=tool_pool_path,
        invocation_examples_path=invocation_examples_path
    )

    # Get available tools
    tools = tool_manager.get_tools_json_schema()
    print(f"Available tools ({len(tools)}):")
    for tool in tools[:5]:
        print(f" - {tool['name']}: {tool['description'][:50]}...")
    if len(tools) > 5:
        print(f" ... and {len(tools) - 5} more")

    # Show Python implementation stats
    python_tools = [t['name'] for t in tools if tool_manager.has_python_implementation(t['name'])]
    print(f"\nTools with Python implementations: {len(python_tools)}/{len(tools)}")

    # Test invoking a Python-implemented tool
    test_tools = [
        ("add", {"a": 10, "b": 20}),
        ("create_ticket", {"title": "Test ticket", "priority": 3}),
        ("get_flight_cost", {"travel_from": "SFO", "travel_to": "JFK", "travel_date": "2024-06-15", "travel_class": "economy"}),
    ]
    for test_tool, test_params in test_tools:
        print(f"\nTesting Python tool: {test_tool}")
        try:
            result = tool_manager.invoke_tool(test_tool, test_params)
            print(f"Result: {result}")
        except Exception as e:
            print(f"Error: {e}")