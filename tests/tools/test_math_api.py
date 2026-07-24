import pytest
import json
import math
from tools.math_api import MathAPI


@pytest.fixture
def math_api():
    initial_config = {
        "numbers": [275.5, 299.75, 250.65, 310.85, 290.1]
    }
    return MathAPI(initial_config)


def test_absolute_value_positive(math_api):
    result = math_api.absolute_value(number=42.5)
    assert result["result"] == 42.5


def test_absolute_value_negative(math_api):
    result = math_api.absolute_value(number=-42.5)
    assert result["result"] == 42.5


def test_absolute_value_zero(math_api):
    result = math_api.absolute_value(number=0.0)
    assert result["result"] == 0.0


def test_add_positive_numbers(math_api):
    result = math_api.add(a=10.5, b=20.3)
    assert result["result"] == pytest.approx(30.8)


def test_add_negative_numbers(math_api):
    result = math_api.add(a=-5.0, b=-10.0)
    assert result["result"] == pytest.approx(-15.0)


def test_add_mixed_numbers(math_api):
    result = math_api.add(a=15.0, b=-5.0)
    assert result["result"] == pytest.approx(10.0)


def test_divide_normal(math_api):
    result = math_api.divide(a=10.0, b=2.0)
    assert result["result"] == pytest.approx(5.0)


def test_divide_by_zero(math_api):
    result = math_api.divide(a=10.0, b=0.0)
    assert result["result"] == float('inf')


def test_divide_negative(math_api):
    result = math_api.divide(a=-10.0, b=2.0)
    assert result["result"] == pytest.approx(-5.0)


def test_imperial_si_conversion_inches_to_cm(math_api):
    result = math_api.imperial_si_conversion(value=10.0, unit_in="inch", unit_out="cm")
    assert result["result"] == pytest.approx(25.4)


def test_imperial_si_conversion_pounds_to_kg(math_api):
    result = math_api.imperial_si_conversion(value=1.0, unit_in="pound", unit_out="kg")
    assert result["result"] == pytest.approx(0.453592, rel=1e-3)


def test_imperial_si_conversion_invalid_unit(math_api):
    result = math_api.imperial_si_conversion(value=10.0, unit_in="invalid", unit_out="cm")
    assert result["result"] == 0.0


def test_logarithm_base_10(math_api):
    result = math_api.logarithm(value=100.0, base=10.0, precision=2)
    assert result["result"] == 2.0


def test_logarithm_custom_base_and_precision(math_api):
    result = math_api.logarithm(value=36.0, base=6.0, precision=4)
    assert result["result"] == pytest.approx(2.0, abs=1e-4)


def test_logarithm_invalid_input(math_api):
    result = math_api.logarithm(value=-10.0, base=10.0, precision=2)
    assert result["result"] == 0.0


def test_max_value_normal(math_api):
    result = math_api.max_value(numbers=[3.0, 16.0, 60.0, -5.0])
    assert result["result"] == 60.0


def test_max_value_single_element(math_api):
    result = math_api.max_value(numbers=[37.0])
    assert result["result"] == 37.0


def test_max_value_empty_list(math_api):
    result = math_api.max_value(numbers=[])
    assert result["result"] == 0.0


def test_mean_normal(math_api):
    result = math_api.mean(numbers=[3.0, 16.0, 60.0])
    assert result["result"] == pytest.approx(26.333333, rel=1e-4)


def test_mean_single_element(math_api):
    result = math_api.mean(numbers=[37.0])
    assert result["result"] == 37.0


def test_mean_empty_list(math_api):
    result = math_api.mean(numbers=[])
    assert result["result"] == 0.0


def test_min_value_normal(math_api):
    result = math_api.min_value(numbers=[3.0, 16.0, 60.0, -5.0])
    assert result["result"] == -5.0


def test_min_value_single_element(math_api):
    result = math_api.min_value(numbers=[37.0])
    assert result["result"] == 37.0


def test_min_value_empty_list(math_api):
    result = math_api.min_value(numbers=[])
    assert result["result"] == 0.0


def test_multiply_positive_numbers(math_api):
    result = math_api.multiply(a=4.0, b=5.0)
    assert result["result"] == pytest.approx(20.0)


def test_multiply_by_zero(math_api):
    result = math_api.multiply(a=100.0, b=0.0)
    assert result["result"] == pytest.approx(0.0)


def test_multiply_negative_numbers(math_api):
    result = math_api.multiply(a=-3.0, b=-7.0)
    assert result["result"] == pytest.approx(21.0)


def test_percentage_normal(math_api):
    result = math_api.percentage(part=25.0, whole=200.0)
    assert result["result"] == pytest.approx(12.5)


def test_percentage_whole_is_zero(math_api):
    result = math_api.percentage(part=25.0, whole=0.0)
    assert result["result"] == 0.0


def test_percentage_part_greater_than_whole(math_api):
    result = math_api.percentage(part=150.0, whole=100.0)
    assert result["result"] == pytest.approx(150.0)


def test_power_integer_exponent(math_api):
    result = math_api.power(base=2.0, exponent=3.0)
    assert result["result"] == pytest.approx(8.0)


def test_power_fractional_exponent(math_api):
    result = math_api.power(base=9.0, exponent=0.5)
    assert result["result"] == pytest.approx(3.0)


def test_power_zero_base_zero_exponent(math_api):
    result = math_api.power(base=0.0, exponent=0.0)
    assert result["result"] == 1.0


def test_round_number_default(math_api):
    result = math_api.round_number(number=3.14159)
    assert result["result"] == 3.0


def test_round_number_custom_decimal(math_api):
    result = math_api.round_number(number=3.14159, decimal_places=2)
    assert result["result"] == 3.14


def test_round_number_negative_decimal(math_api):
    result = math_api.round_number(number=1234.0, decimal_places=-2)
    assert result["result"] == 1200.0


def test_si_unit_conversion_m_to_cm(math_api):
    result = math_api.si_unit_conversion(value=1.0, unit_in="meter", unit_out="centimeter")
    assert result["result"] == pytest.approx(100.0)

def test_si_unit_conversion_kg_to_g(math_api):
    result = math_api.si_unit_conversion(value=2.5, unit_in="kilogram", unit_out="gram")
    assert result["result"] == pytest.approx(2500.0)


def test_si_unit_conversion_invalid_unit(math_api):
    result = math_api.si_unit_conversion(value=10.0, unit_in="invalid", unit_out="g")
    assert result["result"] == 0.0


def test_square_root_normal(math_api):
    result = math_api.square_root(number=16.0, precision=2)
    assert result["result"] == pytest.approx(4.0, abs=1e-2)


def test_square_root_precision(math_api):
    result = math_api.square_root(number=2.0, precision=4)
    assert result["result"] == pytest.approx(1.4142, abs=1e-4)


def test_square_root_negative(math_api):
    result = math_api.square_root(number=-4.0, precision=2)
    assert result["result"] == 0.0


def test_standard_deviation_normal(math_api):
    result = math_api.standard_deviation(numbers=[100, 95, 85, 90, 88, 92])
    mean_val = sum([100, 95, 85, 90, 88, 92]) / 6
    variance = sum((x - mean_val) ** 2 for x in [100, 95, 85, 90, 88, 92]) / 6
    expected = math.sqrt(variance)
    assert result["result"] == pytest.approx(expected, rel=1e-4)


def test_standard_deviation_single_element(math_api):
    result = math_api.standard_deviation(numbers=[50.0])
    assert result["result"] == 0.0


def test_standard_deviation_empty_list(math_api):
    result = math_api.standard_deviation(numbers=[])
    assert result["result"] == 0.0


def test_subtract_positive_numbers(math_api):
    result = math_api.subtract(a=20.0, b=10.0)
    assert result["result"] == pytest.approx(10.0)


def test_subtract_negative_result(math_api):
    result = math_api.subtract(a=5.0, b=10.0)
    assert result["result"] == pytest.approx(-5.0)


def test_subtract_negative_numbers(math_api):
    result = math_api.subtract(a=-5.0, b=-10.0)
    assert result["result"] == pytest.approx(5.0)


def test_sum_values_normal(math_api):
    result = math_api.sum_values(numbers=[1.5, 2.5, 3.0])
    assert result["result"] == pytest.approx(7.0)


def test_sum_values_empty_list(math_api):
    result = math_api.sum_values(numbers=[])
    assert result["result"] == 0.0


def test_sum_values_negative_numbers(math_api):
    result = math_api.sum_values(numbers=[-1.5, -2.5, 4.0])
    assert result["result"] == pytest.approx(0.0)
