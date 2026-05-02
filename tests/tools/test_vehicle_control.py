import pytest
import json
from tools.vehicle_control import VehicleControl


@pytest.fixture
def vehicle():
    initial_config = {
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
    return VehicleControl(initial_config)


class TestActivateParkingBrake:
    def test_engage_normal(self, vehicle):
        result = vehicle.activateParkingBrake(mode='engage')
        assert result['parkingBrakeStatus'] == 'engaged'

    def test_release_normal(self, vehicle):
        vehicle.activateParkingBrake(mode='engage')
        result = vehicle.activateParkingBrake(mode='release')
        assert result['parkingBrakeStatus'] == 'released'

    def test_invalid_mode(self, vehicle):
        result = vehicle.activateParkingBrake(mode='invalid_mode')
        assert 'parkingBrakeStatus' in result


class TestAdjustClimateControl:
    def test_set_celsius(self, vehicle):
        result = vehicle.adjustClimateControl(temperature=25.0, unit='celsius', fanSpeed=70, mode='cool')
        assert result['currentTemperature'] == 25.0
        assert result['climateMode'] == 'cool'

    def test_set_fahrenheit(self, vehicle):
        result = vehicle.adjustClimateControl(temperature=72.0, unit='fahrenheit', fanSpeed=80, mode='auto')
        assert result['currentTemperature'] == pytest.approx((72.0 - 32) * 5 / 9)

    def test_invalid_mode(self, vehicle):
        result = vehicle.adjustClimateControl(temperature=20.0, mode='invalid')
        assert 'currentTemperature' in result


class TestDisplayCarStatus:
    def test_display_fuel(self, vehicle):
        result = vehicle.displayCarStatus(option='fuel')
        assert result['status']['fuelLevel'] == 15.0

    def test_display_doors(self, vehicle):
        result = vehicle.displayCarStatus(option='doors')
        assert result['status']['doorStatus'] == {
            "driver": "locked",
            "passenger": "locked",
            "rear_left": "locked",
            "rear_right": "locked"
        }

    def test_display_invalid_option(self, vehicle):
        result = vehicle.displayCarStatus(option='nonexistent')
        assert result['status'] == {}


class TestDisplayLog:
    def test_display_normal(self, vehicle):
        result = vehicle.display_log(messages=["Engine started", "Door locked"])
        assert "Engine started" in result['log']

    def test_empty_messages(self, vehicle):
        result = vehicle.display_log(messages=[])
        assert len(result['log']) == 0

    def test_invalid_input(self, vehicle):
        result = vehicle.display_log(messages="not_a_list")
        assert isinstance(result, dict)


class TestEstimateDistance:
    def test_normal_estimation(self, vehicle):
        result = vehicle.estimate_distance(cityA='94016', cityB='83214')
        assert 'distance' in result
        assert isinstance(result['distance'], (int, float))

    def test_same_city(self, vehicle):
        result = vehicle.estimate_distance(cityA='94016', cityB='94016')
        assert 'distance' in result

    def test_invalid_zipcode(self, vehicle):
        result = vehicle.estimate_distance(cityA='', cityB='83214')
        assert 'distance' in result


class TestEstimateDriveFeasibilityByMileage:
    def test_feasible_distance(self, vehicle):
        result = vehicle.estimate_drive_feasibility_by_mileage(distance=100.0)
        assert result['canDrive'] is True

    def test_infeasible_distance(self, vehicle):
        result = vehicle.estimate_drive_feasibility_by_mileage(distance=10000.0)
        assert result['canDrive'] is False

    def test_negative_distance(self, vehicle):
        result = vehicle.estimate_drive_feasibility_by_mileage(distance=-50.0)
        assert 'canDrive' in result


class TestFillFuelTank:
    def test_normal_fill(self, vehicle):
        result = vehicle.fillFuelTank(fuelAmount=10.0)
        assert result['fuelLevel'] == pytest.approx(25.0)

    def test_overfill(self, vehicle):
        result = vehicle.fillFuelTank(fuelAmount=60.0)
        assert result['fuelLevel'] == 50.0

    def test_negative_fill(self, vehicle):
        result = vehicle.fillFuelTank(fuelAmount=-5.0)
        assert result['fuelLevel'] == 15.0


class TestGallonToLiter:
    def test_normal_conversion(self, vehicle):
        result = vehicle.gallon_to_liter(gallon=30.0)
        assert result['liter'] == pytest.approx(30.0 * 3.78541)

    def test_zero_gallon(self, vehicle):
        result = vehicle.gallon_to_liter(gallon=0.0)
        assert result['liter'] == 0.0

    def test_negative_gallon(self, vehicle):
        result = vehicle.gallon_to_liter(gallon=-10.0)
        assert 'liter' in result


class TestGetZipcodeBasedOnCity:
    def test_known_city(self, vehicle):
        result = vehicle.get_zipcode_based_on_city('San Francisco')
        assert 'zipcode' in result

    def test_unknown_city(self, vehicle):
        result = vehicle.get_zipcode_based_on_city('UnknownCity')
        assert 'zipcode' in result

    def test_empty_city(self, vehicle):
        result = vehicle.get_zipcode_based_on_city('')
        assert 'zipcode' in result


class TestLiterToGallon:
    def test_normal_conversion(self, vehicle):
        result = vehicle.liter_to_gallon(liter=20.0)
        assert result['gallon'] == pytest.approx(20.0 / 3.78541)

    def test_zero_liter(self, vehicle):
        result = vehicle.liter_to_gallon(liter=0.0)
        assert result['gallon'] == 0.0

    def test_negative_liter(self, vehicle):
        result = vehicle.liter_to_gallon(liter=-5.0)
        assert 'gallon' in result


class TestLockDoors:
    def test_unlock_all_doors(self, vehicle):
        result = vehicle.lockDoors(unlock=True, door=['driver', 'passenger', 'rear_left', 'rear_right'])
        assert result['lockStatus'] == 'unlocked'

    def test_lock_specific_doors(self, vehicle):
        vehicle.lockDoors(unlock=True, door=['driver', 'passenger', 'rear_left', 'rear_right'])
        result = vehicle.lockDoors(unlock=False, door=['driver', 'rear_left'])
        assert result['lockStatus'] == 'locked'
        assert vehicle.doorStatus['driver'] == 'locked'
        assert vehicle.doorStatus['passenger'] == 'unlocked'

    def test_invalid_door(self, vehicle):
        result = vehicle.lockDoors(unlock=False, door=['trunk'])
        assert 'lockStatus' in result


class TestPressBrakePedal:
    def test_full_press(self, vehicle):
        result = vehicle.pressBrakePedal(pedalPosition=1.0)
        assert result['brakePedalStatus'] == 'pressed'
        assert result['brakePedalForce'] == pytest.approx(500.0)

    def test_half_press(self, vehicle):
        result = vehicle.pressBrakePedal(pedalPosition=0.5)
        assert result['brakePedalStatus'] == 'pressed'
        assert result['brakePedalForce'] == pytest.approx(250.0)

    def test_invalid_position(self, vehicle):
        result = vehicle.pressBrakePedal(pedalPosition=1.5)
        assert result['brakePedalStatus'] == 'pressed'


class TestSetCruiseControl:
    def test_activate_cruise(self, vehicle):
        result = vehicle.setCruiseControl(speed=65, activate=True, distanceToNextVehicle=100)
        assert result['cruiseStatus'] == 'active'
        assert result['currentSpeed'] == 65

    def test_deactivate_cruise(self, vehicle):
        vehicle.setCruiseControl(speed=65, activate=True, distanceToNextVehicle=100)
        result = vehicle.setCruiseControl(speed=0, activate=False, distanceToNextVehicle=100)
        assert result['cruiseStatus'] == 'inactive'

    def test_invalid_speed(self, vehicle):
        result = vehicle.setCruiseControl(speed=-10, activate=True, distanceToNextVehicle=100)
        assert result['cruiseStatus'] == 'inactive'


class TestSetHeadlights:
    def test_turn_on(self, vehicle):
        result = vehicle.setHeadlights(mode='on')
        assert result['headlightStatus'] == 'on'

    def test_turn_off(self, vehicle):
        vehicle.setHeadlights(mode='on')
        result = vehicle.setHeadlights(mode='off')
        assert result['headlightStatus'] == 'off'

    def test_invalid_mode(self, vehicle):
        result = vehicle.setHeadlights(mode='blink')
        assert 'headlightStatus' in result


class TestSetNavigation:
    def test_set_valid_destination(self, vehicle):
        result = vehicle.set_navigation(destination='2107 Channing Way, Berkeley, CA')
        assert result['status'] == 'navigating to 2107 Channing Way, Berkeley, CA'

    def test_set_another_destination(self, vehicle):
        result = vehicle.set_navigation(destination='456 Oakwood Avenue, Rivermist, 83214')
        assert result['status'] == 'navigating to 456 Oakwood Avenue, Rivermist, 83214'

    def test_empty_destination(self, vehicle):
        result = vehicle.set_navigation(destination='')
        assert result['status'] == 'navigating to '


class TestStartEngine:
    def test_start_mode(self, vehicle):
        result = vehicle.startEngine(ignitionMode='START')
        assert result['engineState'] == 'running'

    def test_stop_mode(self, vehicle):
        vehicle.startEngine(ignitionMode='START')
        result = vehicle.startEngine(ignitionMode='STOP')
        assert result['engineState'] == 'stopped'

    def test_invalid_mode(self, vehicle):
        result = vehicle.startEngine(ignitionMode='INVALID')
        assert result['engineState'] == 'stopped'
