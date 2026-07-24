"""API-state mutation helpers and rollback behavior."""

import json
import re
from typing import Protocol, override

from step_by_step_models import ObjectMap, StateSnapshot, TrajectoryStep
from step_by_step_protocols import (
    StepByStepMixinBase,
    ToolInstance,
    is_object_list,
    is_object_map,
    is_tool_instance,
)

_COMMON_EXTENSIONS = {
    "7z", "a", "avi", "bak", "bmp", "bz2", "c", "cache", "class",
    "conf", "config", "cpp", "css", "csv", "dll", "doc", "docx",
    "dylib", "ear", "exe", "flv", "gif", "go", "gz", "h", "hpp",
    "html", "ico", "jar", "java", "jpeg", "jpg", "js", "json", "lib",
    "log", "md", "mkv", "mov", "mp3", "mp4", "o", "obj", "pdf",
    "png", "ppt", "pptx", "py", "rar", "rs", "sh", "so", "svg",
    "tar", "tmp", "txt", "war", "webm", "webp", "wmv", "xls", "xlsx",
    "xml", "xz", "yaml", "yml", "zip",
}


def _instance_value(
    instance: ToolInstance,
    name: str,
    default: object = None,
) -> object:
    try:
        return instance.__getattribute__(name)
    except AttributeError:
        return default


class StateManagementMixin(StepByStepMixinBase, Protocol):
    @staticmethod
    def _set_nested_field(
        target: ToolInstance | ObjectMap,
        field_path: str,
        value: object,
    ) -> None:
        """Set a dotted field, creating intermediate mappings as needed."""
        bracket_pattern = re.compile(r"\[('[^']+'|\"[^\"]+\")\]")
        processed_parts: list[str] = []
        for part in bracket_pattern.split(field_path):
            if part:
                processed_parts.extend(part.split("."))

        parts: list[str] = []
        index = 0
        while index < len(processed_parts):
            part = processed_parts[index]
            if (
                index + 1 < len(processed_parts)
                and processed_parts[index + 1].startswith("[")
            ):
                parts.append(part + processed_parts[index + 1])
                index += 2
            else:
                parts.append(part)
                index += 1

        merged_parts: list[str] = []
        for part in parts:
            if (
                merged_parts
                and part.lower() in _COMMON_EXTENSIONS
                and len(part) <= 5
            ):
                merged_parts[-1] += f".{part}"
            else:
                merged_parts.append(part)
        if not merged_parts:
            raise ValueError("field path cannot be empty")

        current: object = target
        for part in merged_parts[:-1]:
            if is_object_map(current):
                current = current.setdefault(part, {})
            elif is_tool_instance(current):
                nested = _instance_value(current, part, {})
                current.__setattr__(part, nested)
                current = nested
            else:
                raise TypeError(f"{part!r} has no mutable attributes")

        last_key = merged_parts[-1]
        if is_object_map(current):
            current[last_key] = value
        elif is_tool_instance(current):
            current.__setattr__(last_key, value)
        else:
            raise TypeError(f"parent of {last_key!r} is not mutable")

    @override
    def _apply_state_modifications(self, modifications: ObjectMap) -> int:
        """Apply requested modifications to live Python tool instances."""
        applied = 0
        instances = self.tool_manager.python_tool_instances

        for class_key, field_changes in modifications.items():
            actual_class_key = class_key
            extra_prefix = ""
            if class_key not in instances and "." in class_key:
                potential_key, _, suffix = class_key.partition(".")
                if potential_key in instances:
                    actual_class_key = potential_key
                    extra_prefix = f"{suffix}."
                    print(
                        f"   ℹ Flat key detected: '{class_key}' -> class='{actual_class_key}', prefix='{extra_prefix}'"
                    )

            instance = instances.get(actual_class_key)
            if not is_tool_instance(instance):
                print(f" ⚠ Unknown class_key: {class_key}, skipping")
                continue

            if not is_object_map(field_changes):
                field_path = extra_prefix.rstrip(".") or class_key
                self._set_nested_field(instance, field_path, field_changes)
                applied += 1
                print(f"   {actual_class_key}.{field_path}: {field_changes}")
                continue

            for field_path, value in field_changes.items():
                effective_path = f"{extra_prefix}{field_path}".rstrip(".")
                try:
                    if (
                        field_path == "current_dir"
                        and class_key == "gorilla_file_system"
                    ):
                        print("   ⚠ Skipping gorilla_file_system.current_dir modification (must use cd tool)")
                        continue
                    if field_path.startswith("APPEND:"):
                        list_field = field_path.removeprefix("APPEND:")
                        current_list = _instance_value(instance, list_field)
                        if is_object_list(current_list):
                            current_list.append(value)
                            applied += 1
                            print(f"   {class_key}.{list_field}: appended item")
                        else:
                            print(f"   ⚠ {class_key}.{list_field} is not a list, skipping append")
                        continue
                    if field_path.startswith("EXTEND:"):
                        list_field = field_path.removeprefix("EXTEND:")
                        current_list = _instance_value(instance, list_field)
                        if is_object_list(current_list) and is_object_list(value):
                            current_list.extend(value)
                            applied += 1
                            print(f"   {class_key}.{list_field}: extended with {len(value)} items")
                        else:
                            print(f"   ⚠ {class_key}.{list_field} extend failed (not list or value not list)")
                        continue

                    self._set_nested_field(instance, effective_path, value)
                    applied += 1
                    rendered = json.dumps(value, default=str)[:100]
                    print(f"   {actual_class_key}.{effective_path}: set to {rendered}")
                except (AttributeError, KeyError, TypeError, ValueError) as exc:
                    print(f"   ⚠ Failed to apply {actual_class_key}.{effective_path}: {exc}")

        return applied

    @override
    def _replay_state(
        self,
        trajectory: list[TrajectoryStep],
        baseline_state: StateSnapshot | None = None,
    ) -> None:
        """Restore a snapshot, or replay calls from the cached configuration."""
        if baseline_state is not None:
            self.tool_manager.restore_api_state(baseline_state)
            return

        self.tool_manager.initialize_api_state(force_new=False)
        for step in trajectory:
            for tool_call in step.tool_calls:
                if self.tool_manager.has_python_implementation(tool_call.tool_name):
                    _ = self.tool_manager.invoke_python_tool(
                        tool_call.tool_name,
                        tool_call.arguments,
                    )

        state = self.tool_manager.get_api_state()
        message_state = state.get("message_api", {})
        if "message_api" not in state or "current_user" in message_state:
            return
        message_instance = self.tool_manager.python_tool_instances.get("message_api")
        if not is_tool_instance(message_instance):
            return
        for step in trajectory:
            for tool_call in step.tool_calls:
                user_id = tool_call.arguments.get("user_id")
                if tool_call.tool_name == "message_login" and user_id is not None:
                    message_instance.__setattr__("current_user", user_id)
                    return
