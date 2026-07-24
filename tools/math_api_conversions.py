"""Unit-conversion operations for MathAPI."""

from typing import NotRequired, TypedDict


class ConversionResult(TypedDict):
    """Result from a unit conversion."""

    result: float
    error: NotRequired[str]


class MathAPIConversionsMixin:
    """Provide imperial and SI unit conversions."""

    def imperial_si_conversion(
        self, value: float, unit_in: str, unit_out: str
    ) -> ConversionResult:
        """
        Convert a value between imperial and SI units.
        """
        unit_in = unit_in.lower().strip()
        unit_out = unit_out.lower().strip()
        imperial_aliases = {"f": "fahrenheit", "c": "celsius"}
        unit_in = imperial_aliases.get(unit_in, unit_in)
        unit_out = imperial_aliases.get(unit_out, unit_out)
        plural_to_singular = {
            "miles": "mile",
            "kilometers": "km",
            "km": "km",
            "pounds": "pound",
            "kilograms": "kg",
            "kg": "kg",
            "inches": "inch",
            "centimeters": "cm",
            "cm": "cm",
            "gallons": "gallon",
            "liters": "liter",
            "liter": "liter",
            "feet": "foot",
            "meters": "meter",
            "meter": "meter",
            "yards": "yard",
            "ounce": "ounce",
            "ounces": "ounce",
            "gram": "gram",
            "grams": "gram",
        }
        unit_in = plural_to_singular.get(unit_in, unit_in)
        unit_out = plural_to_singular.get(unit_out, unit_out)
        conversion_factors = {
            "inch_to_cm": 2.54,
            "cm_to_inch": 1 / 2.54,
            "pound_to_kg": 0.453592,
            "kg_to_pound": 1 / 0.453592,
            "mile_to_km": 1.60934,
            "km_to_mile": 1 / 1.60934,
            "gallon_to_liter": 3.78541,
            "liter_to_gallon": 1 / 3.78541,
            "foot_to_meter": 0.3048,
            "meter_to_foot": 1 / 0.3048,
            "yard_to_meter": 0.9144,
            "meter_to_yard": 1 / 0.9144,
            "ounce_to_gram": 28.3495,
            "gram_to_ounce": 1 / 28.3495,
            "fahrenheit_to_celsius": None,
            "celsius_to_fahrenheit": None,
        }

        key = f"{unit_in}_to_{unit_out}"
        if key in conversion_factors:
            factor = conversion_factors[key]
            if factor is not None:
                return {"result": float(value * factor)}
            if key == "fahrenheit_to_celsius":
                return {"result": float((value - 32) * 5 / 9)}
            if key == "celsius_to_fahrenheit":
                return {"result": float((value * 9 / 5) + 32)}

        if unit_in == unit_out:
            return {"result": float(value)}

        return {"result": 0.0}

    def si_unit_conversion(
        self, value: float, unit_in: str, unit_out: str
    ) -> ConversionResult:
        """
        Convert a value from one SI unit to another.
        """
        si_prefixes = {
            "pico": 1e-12,
            "nano": 1e-9,
            "micro": 1e-6,
            "milli": 1e-3,
            "centi": 1e-2,
            "deci": 1e-1,
            "": 1.0,
            "deca": 1e1,
            "hecto": 1e2,
            "kilo": 1e3,
            "mega": 1e6,
            "giga": 1e9,
            "tera": 1e12,
        }

        base_units = [
            "meter",
            "gram",
            "liter",
            "second",
            "ampere",
            "kelvin",
            "mole",
            "candela",
            "byte",
            "bit",
        ]
        unit_aliases = {
            "m": "meter",
            "g": "gram",
            "l": "liter",
            "s": "second",
            "a": "ampere",
            "k": "kelvin",
            "b": "byte",
            "km": "kilometer",
            "cm": "centimeter",
            "mm": "millimeter",
            "kg": "kilogram",
            "mg": "milligram",
            "ml": "milliliter",
            "cl": "centiliter",
            "ms": "millisecond",
            "us": "microsecond",
            "ns": "nanosecond",
            "ma": "milliampere",
            "kb": "kilobyte",
            "mb": "megabyte",
            "gb": "gigabyte",
        }

        def parse_unit(unit_str: str) -> tuple[float, str]:
            unit_str = unit_str.lower().strip()
            plural_to_singular = {
                "meters": "meter",
                "grams": "gram",
                "liters": "liter",
                "seconds": "second",
                "amperes": "ampere",
                "kelvins": "kelvin",
                "moles": "mole",
                "candelas": "candela",
                "bytes": "byte",
                "bits": "bit",
                "centimeters": "centimeter",
                "millimeters": "millimeter",
                "kilometers": "kilometer",
                "kilograms": "kilogram",
                "milligrams": "milligram",
                "milliliters": "milliliter",
                "centiliters": "centiliter",
                "milliseconds": "millisecond",
                "microseconds": "microsecond",
                "nanoseconds": "nanosecond",
                "kibibytes": "kilobyte",
                "mebibytes": "megabyte",
                "gibibytes": "gigabyte",
                "kilobytes": "kilobyte",
                "megabytes": "megabyte",
                "gigabytes": "gigabyte",
            }
            unit_str = plural_to_singular.get(unit_str, unit_str)
            if unit_str in unit_aliases:
                val = unit_aliases[unit_str]
                for prefix, factor in si_prefixes.items():
                    if prefix:
                        for base in base_units:
                            if val == prefix + base:
                                return factor, base
                return 1.0, val
            for prefix, factor in si_prefixes.items():
                if prefix:
                    for base in base_units:
                        if unit_str == prefix + base:
                            return factor, base
            return 1.0, unit_str

        factor_in, base_in = parse_unit(unit_in)
        factor_out, base_out = parse_unit(unit_out)

        if base_in != base_out:
            return {
                "error": (
                    "Cannot convert between incompatible units: "
                    f"{unit_in} (base: {base_in}) and "
                    f"{unit_out} (base: {base_out})"
                ),
                "result": 0.0,
            }

        return {"result": float(value * factor_in / factor_out)}
