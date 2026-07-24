"""Placeholder resolution for generated tool arguments."""

import copy
from typing import Protocol, override

from step_by_step_models import ObjectMap
from step_by_step_protocols import (
    StepByStepMixinBase,
    is_object_list,
    is_object_map,
    placeholder_keys,
)


class PlaceholderProcessingMixin(StepByStepMixinBase, Protocol):
    @override
    def _process_placeholders(
        self,
        arguments: ObjectMap,
        execution_context: ObjectMap,
    ) -> ObjectMap:
        processed_args = copy.deepcopy(arguments)

        def resolve_placeholder(key_path: str) -> object:
            """Resolve a placeholder key, supporting TURN{N} references."""
            keys = key_path.split(".")
            current: object

            if keys[0].startswith("TURN"):
                turn_num = int(keys[0].removeprefix("TURN")) - 1
                turn_outputs = execution_context.get("turn_outputs", [])
                if not is_object_list(turn_outputs) or turn_num >= len(turn_outputs):
                    return None
                current = turn_outputs[turn_num]
                keys = keys[1:]
            else:
                current = execution_context

            for key in keys:
                if not is_object_map(current) or key not in current:
                    return None
                current = current[key]
            return current

        for arg_name, arg_value in processed_args.items():
            if not isinstance(arg_value, str):
                continue
            for placeholder_key in placeholder_keys(arg_value):
                resolved_value = resolve_placeholder(placeholder_key)
                if resolved_value is None:
                    continue
                placeholder_tag = "{{" + placeholder_key + "}}"
                processed_args[arg_name] = (
                    resolved_value
                    if arg_value == placeholder_tag
                    else arg_value.replace(placeholder_tag, str(resolved_value))
                )
        return processed_args
