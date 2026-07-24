"""Arithmetic and statistical operations for MathAPI."""

import math
from typing import TypedDict


class MathResult(TypedDict):
    """Result from a scalar math operation."""

    result: float


class MathAggregateResult(MathResult):
    """Result from an aggregate math operation."""

    input_numbers: list[float]


class MathAPIOperationsMixin:
    """Provide scalar arithmetic and aggregate operations."""

    def absolute_value(self, number: float) -> MathResult:
        """
        Calculate the absolute value of a number.
        """
        return {"result": float(abs(number))}

    def add(self, a: float, b: float) -> MathResult:
        """
        Add two numbers.
        """
        return {"result": float(a + b)}

    def divide(self, a: float, b: float) -> MathResult:
        """
        Divide one number by another.
        """
        if b == 0:
            return {"result": float("inf") if a >= 0 else float("-inf")}
        return {"result": float(a / b)}

    def logarithm(
        self, value: float, base: float, precision: int
    ) -> MathResult:
        """
        Compute the logarithm of a number with adjustable precision using mpmath.
        """
        if value <= 0 or base <= 0 or base == 1:
            return {"result": 0.0}
        result = math.log(value, base)
        precision = max(0, precision)
        return {"result": float(round(result, precision))}

    def max_value(self, numbers: list[float]) -> MathAggregateResult:
        """
        Find the maximum value in a list of numbers.
        """
        if not numbers:
            return {"result": 0.0, "input_numbers": []}
        return {"result": float(max(numbers)), "input_numbers": list(numbers)}

    def mean(self, numbers: list[float]) -> MathAggregateResult:
        """
        Calculate the mean of a list of numbers.
        """
        if not numbers:
            return {"result": 0.0, "input_numbers": []}
        return {
            "result": float(sum(numbers) / len(numbers)),
            "input_numbers": list(numbers),
        }

    def min_value(self, numbers: list[float]) -> MathAggregateResult:
        """
        Find the minimum value in a list of numbers.
        """
        if not numbers:
            return {"result": 0.0, "input_numbers": []}
        return {"result": float(min(numbers)), "input_numbers": list(numbers)}

    def multiply(self, a: float, b: float) -> MathResult:
        """
        Multiply two numbers.
        """
        return {"result": float(a * b)}

    def percentage(self, part: float, whole: float) -> MathResult:
        """
        Calculate the percentage of a part relative to a whole.
        """
        if whole == 0:
            return {"result": 0.0}
        return {"result": float((part / whole) * 100)}

    def power(self, base: float, exponent: float) -> MathResult:
        """
        Raise a number to a power.
        """
        try:
            return {"result": float(math.pow(base, exponent))}
        except (OverflowError, ValueError):
            return {"result": 0.0}

    def round_number(
        self, number: float, decimal_places: int = 0
    ) -> MathResult:
        """
        Round a number to a specified number of decimal places.
        """
        return {"result": float(round(number, decimal_places))}

    def square_root(self, number: float, precision: int) -> MathResult:
        """
        Calculate the square root of a number with adjustable precision using the decimal module.
        """
        if number < 0:
            return {"result": 0.0}
        result = math.sqrt(number)
        precision = max(0, precision)
        return {"result": float(round(result, precision))}

    def standard_deviation(
        self, numbers: list[float]
    ) -> MathAggregateResult:
        """
        Calculate the standard deviation of a list of numbers.
        """
        if not numbers:
            return {"result": 0.0, "input_numbers": []}
        if len(numbers) == 1:
            return {"result": 0.0, "input_numbers": list(numbers)}
        mean_val = sum(numbers) / len(numbers)
        variance = sum((x - mean_val) ** 2 for x in numbers) / len(numbers)
        return {
            "result": float(math.sqrt(variance)),
            "input_numbers": list(numbers),
        }

    def subtract(self, a: float, b: float) -> MathResult:
        """
        Subtract one number from another.
        """
        return {"result": float(a - b)}

    def sum_values(self, numbers: list[float]) -> MathAggregateResult:
        """
        Calculate the sum of a list of numbers.
        """
        if not numbers:
            return {"result": 0.0, "input_numbers": []}
        return {"result": float(sum(numbers)), "input_numbers": list(numbers)}
