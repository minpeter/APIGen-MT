"""Typed post-processing helpers for prepared API state."""

from step_by_step_models import QueryGenerationResult
from step_by_step_protocols import (
    StepByStepToolManager,
    ToolInstance,
    is_object_list,
    is_object_map,
    is_tool_instance,
)

MESSAGE_AUTH_TOOLS = {
    "add_contact",
    "delete_message",
    "get_user_id",
    "search_messages",
    "send_message",
}
FILE_TOOLS = {
    "cat", "cp", "diff", "echo", "grep", "mv", "rm", "rmdir", "sort",
    "tail", "touch", "wc",
}
POTENTIAL_FILES = [
    "config.json",
    "processor.py",
    "README.md",
    "data.json",
    "temp.txt",
]


def instance_value(
    instance: ToolInstance,
    name: str,
    default: object = None,
) -> object:
    try:
        return instance.__getattribute__(name)
    except AttributeError:
        return default


def _find_file(node: object, target: str) -> object:
    if not is_object_map(node):
        return None
    if target in node:
        return node[target]
    for value in node.values():
        found = _find_file(value, target)
        if found is not None:
            return found
    return None


def _string_list(value: object) -> list[str] | None:
    if not is_object_list(value) or not all(
        isinstance(item, str) for item in value
    ):
        return None
    return [item for item in value if isinstance(item, str)]


def prepare_message_state(
    manager: StepByStepToolManager,
    query_result: QueryGenerationResult,
    applied: int,
) -> int:
    if applied <= 0:
        return applied
    message_api = manager.python_tool_instances.get("message_api")
    if not is_tool_instance(message_api):
        return applied

    user_map = instance_value(message_api, "user_map", {})
    if not is_object_map(user_map):
        return applied

    ids_to_names: dict[str, str] = {}
    names_to_remove: set[str] = set()
    for name, user_id in list(user_map.items()):
        if not isinstance(user_id, str):
            continue
        existing_name = ids_to_names.get(user_id)
        if existing_name is not None and existing_name != name:
            names_to_remove.add(name)
            print(
                f"   [DEDUP] Removing duplicate user_map entry '{name}' -> {user_id} (conflicts with '{existing_name}' -> {user_id})"
            )
        else:
            ids_to_names[user_id] = name
    for name in names_to_remove:
        del user_map[name]
        applied -= 1

    needs_auth = any(
        tool in MESSAGE_AUTH_TOOLS
        for tool in query_result.expected_tools
    )
    if needs_auth and not instance_value(message_api, "current_user"):
        first_user_id = next(iter(user_map.values()), None)
        if first_user_id:
            message_api.__setattr__("current_user", first_user_id)
            applied += 1
            print(
                "   [AUTH FALLBACK] Auto-set current_user to "
                + f"{first_user_id} for message operations"
            )
    return applied


def prepare_file_state(
    manager: StepByStepToolManager,
    query_result: QueryGenerationResult,
    applied: int,
) -> None:
    if applied <= 0:
        return
    file_system = manager.python_tool_instances.get("gorilla_file_system")
    if not is_tool_instance(file_system):
        return
    current_dir = _string_list(instance_value(file_system, "current_dir"))
    root = instance_value(file_system, "root")
    if current_dir is None or not is_object_map(root):
        return

    current = root
    for part in current_dir:
        entry = current.setdefault(
            part,
            {"type": "directory", "contents": {}},
        )
        if not is_object_map(entry):
            return
        contents = entry.get("contents")
        current = contents if is_object_map(contents) else entry
    print(f"   [FS FIX] Ensured directory path exists: {'/'.join(current_dir)}")

    if not any(tool in FILE_TOOLS for tool in query_result.expected_tools):
        return
    query_lower = query_result.query.lower()
    for filename in POTENTIAL_FILES:
        if (
            filename not in query_lower
            and filename.replace(".py", "_v1.0.py") not in query_lower
        ):
            continue
        if filename not in current:
            existing_file = _find_file(root, filename)
            if existing_file is not None:
                current[filename] = existing_file
                print(f"   [FS FIX] Moved '{filename}' to current directory")
    print(
        f"   [FS FIX] current_dir={current_dir}, files now in location: {list(current)}"
    )
