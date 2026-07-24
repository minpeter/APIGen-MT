"""Argument and cross-turn consistency validation."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

if TYPE_CHECKING:
    from src.multi_turn_protocols import ExecutionContext, GeneratorMixinBase
    from src.step_by_step_models import TrajectoryStep
else:
    from multi_turn_protocols import GeneratorMixinBase
    from step_by_step_models import TrajectoryStep

from multi_turn_protocols import (
    is_object_dict,
    is_object_list,
    tool_call_view,
)


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""


def _number_list(value: object) -> list[int | float]:
    if not is_object_list(value):
        return []
    return [item for item in value if isinstance(item, int | float)]


class ConsistencyValidationMixin(GeneratorMixinBase):
    """Reject known hallucination and cross-turn mismatch patterns."""

    @staticmethod
    @override
    def _validate_tool_arguments(
        trajectory: list[TrajectoryStep],
    ) -> list[str]:
        """Check tool call arguments and outputs for hallucination indicators."""
        errors: list[str] = []
        for step in trajectory:
            for raw_tool_call in step.tool_calls:
                tool_call = tool_call_view(raw_tool_call)
                arguments = tool_call.arguments
                raw_output = tool_call.output
                output = raw_output if is_object_dict(raw_output) else {}
                name = tool_call.tool_name

                if name == "book_flight":
                    for field in ("travel_date", "travel_to", "travel_from"):
                        value = arguments.get(field)
                        if not value or str(value).strip() == "":
                            errors.append(
                                f"book_flight: hallucinated empty '{field}' in arguments"
                            )
                    booking_history = output.get("booking_history")
                    if is_object_dict(booking_history):
                        has_route = bool(
                            booking_history.get("travel_date")
                            and booking_history.get("travel_to")
                        )
                    else:
                        has_route = False
                    if not has_route:
                        errors.append(
                            "book_flight: output booking_history missing "
                            + "travel_date/travel_to"
                        )
                    if not output.get("booking_id"):
                        errors.append("book_flight: empty booking_id in output")

                elif name == "purchase_insurance":
                    if not arguments.get("booking_id"):
                        errors.append(
                            "purchase_insurance: empty booking_id in arguments"
                        )
                    insurance_id = output.get("insurance_id", "")
                    insurance_status = output.get("insurance_status")
                    if (
                        insurance_id == "" or insurance_id is None
                    ) and insurance_status is False:
                        errors.append(
                            f"purchase_insurance: failed (ins_id='{insurance_id}', "
                            + f"status={insurance_status}), likely operating on "
                            + "cancelled booking"
                        )

                elif name == "retrieve_invoice":
                    invoice = output.get("invoice")
                    if is_object_dict(invoice) and not invoice:
                        errors.append(
                            "retrieve_invoice: empty invoice dict in output"
                        )

                elif name == "cancel_booking":
                    cancel_status = output.get("cancel_status")
                    if not cancel_status and cancel_status is not None:
                        errors.append("cancel_booking: cancel_status=False")

                elif name == "authenticate_travel":
                    success = output.get("success")
                    if not success and success is not None:
                        error = _text(output.get("error"))
                        errors.append(
                            f"authenticate_travel: failed success={success} "
                            + f"error={error[:60]}"
                        )

                elif name == "get_flight_cost":
                    error = _text(output.get("error"))
                    if error:
                        errors.append(f"get_flight_cost: error={error[:80]}")

                elif name in {
                    "ls", "cat", "cd", "mkdir", "mv", "rm", "rmdir", "touch",
                    "cp", "grep", "find", "wc", "tail", "echo", "du", "sort",
                } and "calls" in arguments:
                    errors.append(
                        f"{name}: LLM generated 'calls' batch format - use single "
                        + "tool call with direct arguments"
                    )

        return errors

    @override
    def _validate_cross_turn_consistency(
        self,
        trajectory: list[TrajectoryStep],
        execution_context: ExecutionContext,
    ) -> list[str]:
        """Validate that tool calls are consistent with prior turn outputs."""
        current_calls = {
            tool_call.tool_name: tool_call
            for step in trajectory
            for raw_tool_call in step.tool_calls
            for tool_call in (tool_call_view(raw_tool_call),)
        }
        prior_outputs: dict[str, list[dict[object, object]]] = {}
        raw_turn_outputs = execution_context.get("turn_outputs")
        if is_object_list(raw_turn_outputs):
            for raw_turn_output in raw_turn_outputs:
                if not is_object_dict(raw_turn_output):
                    continue
                for raw_name, raw_output in raw_turn_output.items():
                    if isinstance(raw_name, str) and is_object_dict(raw_output):
                        prior_outputs.setdefault(raw_name, []).append(raw_output)

        errors: list[str] = []
        booking_call = current_calls.get("book_flight")
        if booking_call is not None:
            arguments = booking_call.arguments
            travel_from = _text(arguments.get("travel_from")).upper()
            travel_to = _text(arguments.get("travel_to")).upper()

            if cost_outputs := prior_outputs.get("get_flight_cost"):
                cost_output = cost_outputs[-1]
                cost_from = _text(cost_output.get("travel_from")).upper()
                cost_to = _text(cost_output.get("travel_to")).upper()
                if cost_from and cost_to and (
                    travel_from != cost_from or travel_to != cost_to
                ):
                    errors.append(
                        "book_flight: route mismatch. get_flight_cost used "
                        + f"{cost_from}→{cost_to} but book_flight called with "
                        + f"{travel_from}→{travel_to}"
                    )

            if airport_outputs := prior_outputs.get(
                "get_nearest_airport_by_city"
            ):
                prior_airports = {
                    nearest.upper()
                    for output in airport_outputs
                    if (nearest := _text(output.get("nearest_airport")))
                }
                if prior_airports and travel_from not in prior_airports:
                    errors.append(
                        f"book_flight: travel_from='{travel_from}' not in prior "
                        + f"airport lookups {prior_airports}"
                    )

        insurance_call = current_calls.get("purchase_insurance")
        if insurance_call is not None:
            booking_id = _text(insurance_call.arguments.get("booking_id"))
            raw_output = insurance_call.output
            output = raw_output if is_object_dict(raw_output) else {}
            if booking_outputs := prior_outputs.get("book_flight"):
                prior_booking_ids = {
                    prior_id
                    for booking_output in booking_outputs
                    if (prior_id := _text(booking_output.get("booking_id")))
                }
                if (
                    prior_booking_ids
                    and booking_id
                    and booking_id not in prior_booking_ids
                ):
                    errors.append(
                        f"purchase_insurance: booking_id='{booking_id}' not in "
                        + f"prior bookings {prior_booking_ids}"
                    )
            if output.get("insurance_status") is False:
                errors.append(
                    f"purchase_insurance: failed (booking_id='{booking_id}', "
                    + "status=False)"
                )

        aggregate_chains = {
            "mean": {"min_value", "max_value", "standard_deviation", "sum_values"},
            "min_value": {"mean", "max_value", "standard_deviation", "sum_values"},
            "max_value": {"mean", "min_value", "standard_deviation", "sum_values"},
            "sum_values": {"mean", "min_value", "max_value", "standard_deviation"},
            "standard_deviation": {"mean", "min_value", "max_value", "sum_values"},
        }
        for current_name, current_call in current_calls.items():
            compatible_prior = aggregate_chains.get(current_name)
            if compatible_prior is None:
                continue
            current_numbers = _number_list(
                current_call.arguments.get("numbers")
            )
            for prior_name, outputs in prior_outputs.items():
                if prior_name not in compatible_prior or not outputs:
                    continue
                prior_numbers = _number_list(outputs[-1].get("input_numbers"))
                if (
                    current_numbers
                    and prior_numbers
                    and set(current_numbers) != set(prior_numbers)
                ):
                    errors.append(
                        f"{current_name}: input numbers {current_numbers} do not "
                        + f"match prior {prior_name} inputs {prior_numbers} - query "
                        + "says 'same' values but arguments use different numbers"
                    )

        return errors
