#!/usr/bin/env python3
"""
Test script for the Reactive Layer of the Intelligent Driving Agent.
Simulates critical scenarios and tests immediate response capabilities.
"""

import time
import random
from typing import Dict, List
from dataclasses import dataclass


# Import the reactive layer (or define it inline if not in separate module)
@dataclass
class VehicleState:
    """Simplified vehicle state for testing"""
    speed: float = 0.0  # km/h
    leading_vehicle: Dict = None
    timestamp: float = 0.0


class ReactiveLayer:
    """Ultra-fast reactive layer for critical situations"""

    def __init__(self, config=None):
        if config is None:
            config = {}
        self.critical_ttc = config.get("critical_ttc", 1.2)  # seconds
        self.critical_speed = config.get("critical_speed", 100)  # km/h
        self.min_safe_distance = config.get("min_safe_distance", 5.0)  # meters
        self.response_times = []

    def check_immediate_danger(self, vehicle_state) -> str:
        """
        Check for immediate dangers that require instant reaction
        Returns: alert type or None
        """
        start_time = time.perf_counter()

        # 1. Check Time-to-Collision
        ttc = self._calculate_ttc(vehicle_state)
        if ttc is not None and ttc < self.critical_ttc:
            alert = "emergency_brake"
        # 2. Check excessive speed
        elif vehicle_state.speed > self.critical_speed:
            alert = "speed_warning"
        # 3. Check safe distance
        elif (vehicle_state.leading_vehicle and
              vehicle_state.leading_vehicle.get("distance", 100) < self.min_safe_distance):
            alert = "following_too_close"
        else:
            alert = None

        response_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
        self.response_times.append(response_time)

        return alert

    def _calculate_ttc(self, vehicle_state) -> float:
        """Calculate time to collision with leading vehicle"""
        if not vehicle_state.leading_vehicle:
            return None

        distance = vehicle_state.leading_vehicle.get("distance", 0)
        relative_speed = vehicle_state.speed - vehicle_state.leading_vehicle.get("speed", 0)

        if relative_speed <= 0:
            return float('inf')  # No collision course

        return distance / (relative_speed / 3.6)  # Convert km/h to m/s


class ReactiveLayerTester:
    """Test harness for the reactive layer"""

    def __init__(self):
        self.config = {
            "critical_ttc": 1.5,  # 1.5 seconds
            "critical_speed": 90,  # 90 km/h
            "min_safe_distance": 3.0  # 3 meters
        }
        self.reactive_layer = ReactiveLayer(self.config)
        self.test_scenarios = []

    def create_test_scenarios(self) -> List[Dict]:
        """Create various test scenarios for the reactive layer"""
        scenarios = [
            # Scenario 1: Emergency brake situation (low TTC)
            {
                "name": "Emergency Brake - Low TTC",
                "vehicle_state": VehicleState(
                    speed=80.0,  # 80 km/h
                    leading_vehicle={"distance": 10.0, "speed": 30.0},  # 10m at 30 km/h
                    timestamp=time.time()
                ),
                "expected_alert": "emergency_brake",
                "description": "Vehicle approaching much slower vehicle with only 10m distance"
            },

            # Scenario 2: Critical speed
            {
                "name": "Critical Speed",
                "vehicle_state": VehicleState(
                    speed=95.0,  # Above threshold
                    leading_vehicle={"distance": 50.0, "speed": 90.0},
                    timestamp=time.time()
                ),
                "expected_alert": "speed_warning",
                "description": "Vehicle exceeding 90 km/h speed limit"
            },

            # Scenario 3: Following too close
            {
                "name": "Following Too Close",
                "vehicle_state": VehicleState(
                    speed=60.0,
                    leading_vehicle={"distance": 2.5, "speed": 60.0},  # Only 2.5m at same speed
                    timestamp=time.time()
                ),
                "expected_alert": "following_too_close",
                "description": "Vehicle following at unsafe distance (2.5m)"
            },

            # Scenario 4: Safe situation (no alert expected)
            {
                "name": "Safe Driving",
                "vehicle_state": VehicleState(
                    speed=70.0,
                    leading_vehicle={"distance": 40.0, "speed": 70.0},
                    timestamp=time.time()
                ),
                "expected_alert": None,
                "description": "Safe following distance and speed"
            },

            # Scenario 5: Edge case - very low TTC
            {
                "name": "Extreme Emergency",
                "vehicle_state": VehicleState(
                    speed=50.0,
                    leading_vehicle={"distance": 2.0, "speed": 0.0},  # Stopped vehicle 2m ahead
                    timestamp=time.time()
                ),
                "expected_alert": "emergency_brake",
                "description": "Extremely low TTC (vehicle approaching stopped car)"
            },

            # Scenario 6: No leading vehicle
            {
                "name": "No Leading Vehicle",
                "vehicle_state": VehicleState(
                    speed=85.0,
                    leading_vehicle=None,
                    timestamp=time.time()
                ),
                "expected_alert": None,
                "description": "No vehicle ahead, but speed is safe"
            }
        ]
        return scenarios

    def run_single_test(self, scenario: Dict) -> bool:
        """Run a single test scenario and return pass/fail"""
        print(f"\n{'=' * 60}")
        print(f"Testing: {scenario['name']}")
        print(f"Description: {scenario['description']}")
        print(f"{'-' * 60}")

        # Display scenario details
        state = scenario['vehicle_state']
        print(f"  Vehicle Speed: {state.speed:.1f} km/h")
        if state.leading_vehicle:
            print(f"  Leading Vehicle Distance: {state.leading_vehicle.get('distance', 0):.1f} m")
            print(f"  Leading Vehicle Speed: {state.leading_vehicle.get('speed', 0):.1f} km/h")
            ttc = self.reactive_layer._calculate_ttc(state)
            if ttc is not None and ttc < float('inf'):
                print(f"  Calculated TTC: {ttc:.2f} s")
        else:
            print(f"  No leading vehicle")

        # Run the reactive layer check
        alert = self.reactive_layer.check_immediate_danger(state)

        # Check result
        passed = alert == scenario['expected_alert']

        # Display result
        if alert:
            print(f"  ⚠️  ALERT GENERATED: {alert}")
        else:
            print(f"  ✅ No alert (safe situation)")

        if passed:
            print(f"  ✅ TEST PASSED")
        else:
            print(f"  ❌ TEST FAILED")
            print(f"    Expected: {scenario['expected_alert']}")
            print(f"    Got: {alert}")

        return passed

    def run_performance_test(self, iterations: int = 1000):
        """Test the performance of the reactive layer"""
        print(f"\n{'=' * 60}")
        print(f"PERFORMANCE TEST: {iterations} iterations")
        print(f"{'-' * 60}")

        # Create a typical scenario
        test_state = VehicleState(
            speed=80.0,
            leading_vehicle={"distance": 20.0, "speed": 70.0},
            timestamp=time.time()
        )

        start_time = time.perf_counter()

        for i in range(iterations):
            # Add small variations to simulate real data
            test_state.speed = 80.0 + random.uniform(-5, 5)
            if test_state.leading_vehicle:
                test_state.leading_vehicle["distance"] = 20.0 + random.uniform(-2, 2)

            self.reactive_layer.check_immediate_danger(test_state)

        total_time = time.perf_counter() - start_time
        avg_time = total_time / iterations * 1000  # ms per check

        # Get response time statistics
        if self.reactive_layer.response_times:
            min_time = min(self.reactive_layer.response_times)
            max_time = max(self.reactive_layer.response_times)
            avg_response = sum(self.reactive_layer.response_times) / len(self.reactive_layer.response_times)

            print(f"Total execution time: {total_time * 1000:.2f} ms")
            print(f"Average per check: {avg_time:.3f} ms")
            print(f"Min response time: {min_time:.3f} ms")
            print(f"Max response time: {max_time:.3f} ms")
            print(f"Avg response time: {avg_response:.3f} ms")

            # Check if meets real-time requirements
            if avg_response < 50:  # Our target for reactive layer
                print(f"✅ PERFORMANCE: Meets real-time requirement (< 50 ms)")
            else:
                print(f"⚠️  PERFORMANCE: May not meet real-time requirement")
        else:
            print("No response time data collected")

    def run_stress_test(self):
        """Test with extreme values"""
        print(f"\n{'=' * 60}")
        print(f"STRESS TEST: Extreme Scenarios")
        print(f"{'-' * 60}")

        extreme_scenarios = [
            ("Negative distance", VehicleState(speed=50, leading_vehicle={"distance": -5, "speed": 50})),
            ("Zero speed difference", VehicleState(speed=60, leading_vehicle={"distance": 5, "speed": 60})),
            ("Very high speed", VehicleState(speed=300, leading_vehicle={"distance": 100, "speed": 100})),
            ("Zero distance, same speed", VehicleState(speed=60, leading_vehicle={"distance": 0, "speed": 60})),
        ]

        for name, state in extreme_scenarios:
            try:
                alert = self.reactive_layer.check_immediate_danger(state)
                print(f"{name}: Alert = {alert}")
            except Exception as e:
                print(f"{name}: ERROR - {e}")

    def run_all_tests(self):
        """Run all test suites"""
        print("🚗 REACTIVE LAYER TEST SUITE")
        print("Testing immediate danger detection capabilities")
        print("=" * 60)

        # Create test scenarios
        scenarios = self.create_test_scenarios()

        # Run functional tests
        print("\n📋 FUNCTIONAL TESTS")
        passed_count = 0
        total_count = len(scenarios)

        for scenario in scenarios:
            if self.run_single_test(scenario):
                passed_count += 1

        # Run performance test
        self.run_performance_test(iterations=10000)

        # Run stress test
        self.run_stress_test()

        # Summary
        print(f"\n{'=' * 60}")
        print("📊 TEST SUMMARY")
        print(f"{'-' * 60}")
        print(f"Functional Tests: {passed_count}/{total_count} passed ({passed_count / total_count * 100:.1f}%)")
        print(f"Performance: {len(self.reactive_layer.response_times)} checks executed")

        if passed_count == total_count:
            print(f"\n🎉 ALL FUNCTIONAL TESTS PASSED!")
        else:
            print(f"\n⚠️  SOME TESTS FAILED. Review the results above.")

        # Display configuration
        print(f"\n⚙️  Reactive Layer Configuration:")
        for key, value in self.config.items():
            print(f"  {key}: {value}")


def main():
    """Main execution function"""
    tester = ReactiveLayerTester()
    tester.run_all_tests()

    # Example of real-time simulation
    print(f"\n{'=' * 60}")
    print("🔄 REAL-TIME SIMULATION DEMO")
    print(f"{'-' * 60}")

    # Simulate a dangerous situation unfolding
    print("Simulating a dangerous overtaking scenario...")

    reactive_layer = ReactiveLayer()
    state = VehicleState(speed=85.0, leading_vehicle={"distance": 50.0, "speed": 80.0})

    for i in range(10):
        # Vehicle gets closer each iteration
        state.leading_vehicle["distance"] = 50.0 - (i * 5)
        state.speed = 85.0 + (i * 1)

        alert = reactive_layer.check_immediate_danger(state)

        if alert:
            print(f"  [t={i}] Distance: {state.leading_vehicle['distance']:.1f}m | Speed: {state.speed:.1f}km/h")
            print(f"     ⚠️  {alert.upper()} ALERT! ⚠️")
            if i < 9:  # Don't break on last iteration
                break
        else:
            print(f"  [t={i}] Distance: {state.leading_vehicle['distance']:.1f}m | Speed: {state.speed:.1f}km/h - Safe")

        time.sleep(0.5)  # Simulate real-time delay

    print(f"\n✅ Reactive layer test completed!")


if __name__ == "__main__":
    main()