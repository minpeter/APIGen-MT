from typing import Dict, Any, Optional

TOOL_CLASSES: Dict[str, str] = {
    "gorilla_file_system": "tools.gorilla_file_system",
    "math_api": "tools.math_api",
    "message_api": "tools.message_api",
    "posting_api": "tools.posting_api",
    "ticket_api": "tools.ticket_api",
    "trading_bot": "tools.trading_bot",
    "travel_booking": "tools.travel_booking",
    "vehicle_control": "tools.vehicle_control",
}

_CLASS_NAME_MAP = {
    "gorilla_file_system": "GorillaFileSystem",
    "math_api": "MathAPI",
    "message_api": "MessageAPI",
    "posting_api": "PostingAPI",
    "ticket_api": "TicketAPI",
    "trading_bot": "TradingBot",
    "travel_booking": "TravelBooking",
    "vehicle_control": "VehicleControl",
}


def create_tool_instance(
    class_key: str, initial_config: Optional[Dict[str, Any]] = None
) -> Any:
    import importlib

    if class_key not in TOOL_CLASSES:
        raise ValueError(
            f"Unknown tool class: {class_key}. Available: {list(TOOL_CLASSES.keys())}"
        )

    module_path = TOOL_CLASSES[class_key]
    class_name = _CLASS_NAME_MAP[class_key]
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    config = initial_config or {}
    return cls(initial_config=config)


__all__ = ["TOOL_CLASSES", "create_tool_instance"]
