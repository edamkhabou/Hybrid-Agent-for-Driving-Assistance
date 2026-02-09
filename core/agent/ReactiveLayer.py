"""
Reactive Layer: Fast, rule-based emergency response
"""
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import numpy as np

from config.settings import AGENT_CONFIG, VEHICLE_CONFIG, AgentConfig


@dataclass
class CriticalAlert:
    """Represents a critical alert from reactive layer"""
    type: str
    severity: str  # "warning", "critical", "emergency"
    message: str
    timestamp: float
    reaction_time_ms: float
    parameters: Dict[str, Any] = None

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}

    def to_dict(self) -> Dict:
        return {
            "type": self.type,
            "severity": self.severity,
            "message": self.message,
            "timestamp": self.timestamp,
            "reaction_time_ms": self.reaction_time_ms,
            "parameters": self.parameters
        }


class ReactiveLayer:
    """
    Ultra-fast reactive safety layer.
    Implements reflex-like responses to immediate dangers.
    """

    def __init__(self, config: AgentConfig = None):
        self.config = config or AGENT_CONFIG
        self.alert_history = []
        self.performance_stats = {
            "total_checks": 0,
            "alerts_generated": 0,
            "avg_response_time": 0.0,
            "max_response_time": 0.0
        }
        print("✅ Reactive Layer initialized")

    def check_immediate_danger(self, vehicle_state: Dict) -> Optional[CriticalAlert]:
        """
        Check for immediate dangers (must complete in < 50ms)
        Returns CriticalAlert if danger detected, None otherwise
        """
        start_time = time.perf_counter()
        self.performance_stats["total_checks"] += 1

        alert = None

        # Check each danger type in priority order
        if self._check_emergency_brake(vehicle_state):
            ttc = self._calculate_ttc(vehicle_state)
            alert = CriticalAlert(
                type="emergency_brake",
                severity="emergency",
                message="EMERGENCY: Collision imminent! Brake hard now!",
                timestamp=time.time(),
                reaction_time_ms=0.0,  # will be set later
                parameters={"ttc": ttc, "required_deceleration": self._calculate_required_decel(vehicle_state)}
            )

        elif self._check_critical_speed(vehicle_state):
            alert = CriticalAlert(
                type="critical_speed",
                severity="critical",
                message=f"CRITICAL: Speed {vehicle_state.get('speed', 0):.0f} km/h exceeds safety limit!",
                timestamp=time.time(),
                reaction_time_ms=0.0,
                parameters={"current_speed": vehicle_state.get('speed', 0),
                            "speed_limit": vehicle_state.get('speed_limit', 90)}
            )

        elif self._check_unsafe_following(vehicle_state):
            distance = vehicle_state.get('leading_vehicle', {}).get('distance', 0)
            alert = CriticalAlert(
                type="unsafe_following",
                severity="warning",
                message=f"WARNING: Following too close! Distance: {distance:.1f}m",
                timestamp=time.time(),
                reaction_time_ms=0.0,
                parameters={"following_distance": distance,
                            "min_safe_distance": self._calculate_safe_distance(vehicle_state)}
            )

        elif self._check_sudden_obstacle(vehicle_state):
            alert = CriticalAlert(
                type="sudden_obstacle",
                severity="critical",
                message="CRITICAL: Sudden obstacle detected! Take evasive action!",
                timestamp=time.time(),
                reaction_time_ms=0.0
            )

        # Calculate reaction time
        if alert:
            reaction_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
            alert.reaction_time_ms = reaction_time

            # Update performance stats
            self.alert_history.append(alert)
            self.performance_stats["alerts_generated"] += 1
            self.performance_stats["avg_response_time"] = (
                    self.performance_stats["avg_response_time"] * 0.9 + reaction_time * 0.1
            )
            self.performance_stats["max_response_time"] = max(
                self.performance_stats["max_response_time"], reaction_time
            )

            # Log if response is too slow
            if reaction_time > self.config.MAX_REACTIVE_TIME_MS:
                print(
                    f"⚠️  Warning: Reactive response took {reaction_time:.1f}ms (> {self.config.MAX_REACTIVE_TIME_MS}ms)")

        return alert

    def _check_emergency_brake(self, state: Dict) -> bool:
        """Check if emergency braking is required (TTC < threshold)"""
        ttc = self._calculate_ttc(state)
        if ttc is None or ttc > self.config.CRITICAL_TTC:
            return False

        # Check if deceleration required is beyond vehicle capability
        required_decel = self._calculate_required_decel(state)
        return required_decel > VEHICLE_CONFIG.MAX_DECELERATION * 0.8  # Use 80% of max

    def _check_critical_speed(self, state: Dict) -> bool:
        """Check if speed is critically high"""
        current_speed = state.get('speed', 0)
        speed_limit = state.get('speed_limit', 90)

        # Critical if exceeding speed limit by >20% or absolute threshold
        return (current_speed > self.config.CRITICAL_SPEED or
                current_speed > speed_limit * 1.2)

    def _check_unsafe_following(self, state: Dict) -> bool:
        """Check if following distance is unsafe"""
        if 'leading_vehicle' not in state or not state['leading_vehicle']:
            return False

        distance = state['leading_vehicle'].get('distance', float('inf'))
        safe_distance = self._calculate_safe_distance(state)

        return distance < safe_distance * 0.5  # Less than 50% of safe distance

    def _check_sudden_obstacle(self, state: Dict) -> bool:
        """Check for sudden obstacles (rapid deceleration ahead)"""
        if 'leading_vehicle' not in state:
            return False

        # Check if leading vehicle is decelerating rapidly
        lead_accel = state['leading_vehicle'].get('acceleration', 0)
        return lead_accel < -self.config.EMERGENCY_DECEL_THRESHOLD

    def _calculate_ttc(self, state: Dict) -> Optional[float]:
        """Calculate Time To Collision with leading vehicle"""
        if ('leading_vehicle' not in state or
                not state['leading_vehicle'] or
                'distance' not in state['leading_vehicle']):
            return None

        distance = state['leading_vehicle']['distance']
        ego_speed = state.get('speed', 0) / 3.6  # Convert to m/s
        lead_speed = state['leading_vehicle'].get('speed', 0) / 3.6

        relative_speed = ego_speed - lead_speed

        if relative_speed <= 0:
            return float('inf')  # No collision course

        return distance / relative_speed

    def _calculate_required_decel(self, state: Dict) -> float:
        """Calculate required deceleration to avoid collision"""
        ttc = self._calculate_ttc(state)
        if ttc is None or ttc == float('inf'):
            return 0.0

        ego_speed = state.get('speed', 0) / 3.6  # m/s
        required_decel = ego_speed / ttc if ttc > 0 else float('inf')

        return required_decel

    def _calculate_safe_distance(self, state: Dict) -> float:
        """Calculate safe following distance based on speed"""
        speed_mps = state.get('speed', 0) / 3.6

        # 2-second rule plus vehicle length
        return max(
            self.config.MIN_SAFE_DISTANCE,
            speed_mps * 2 + VEHICLE_CONFIG.VEHICLE_LENGTH
        )

    def get_performance_report(self) -> Dict:
        """Get performance statistics report"""
        return {
            **self.performance_stats,
            "meets_sla": self.performance_stats.get("max_response_time", 0) < self.config.MAX_REACTIVE_TIME_MS,
            "alert_types": {
                alert.type: len([a for a in self.alert_history if a.type == alert.type])
                for alert in self.alert_history[:10]  # Recent alerts only
            }
        }

    def reset_stats(self):
        """Reset performance statistics"""
        self.performance_stats = {
            "total_checks": 0,
            "alerts_generated": 0,
            "avg_response_time": 0.0,
            "max_response_time": 0.0
        }
        self.alert_history = []


# Test function for PyCharm
def test_reactive_layer():
    """Test the reactive layer implementation"""
    print("\n🧪 Testing Reactive Layer...")

    reactive = ReactiveLayer()

    # Test 1: Emergency brake scenario
    print("\nTest 1: Emergency Brake")
    test_state = {
        "speed": 80,  # 80 km/h
        "leading_vehicle": {
            "distance": 10,  # 10 meters
            "speed": 0,  # stopped vehicle
            "acceleration": 0
        }
    }

    alert = reactive.check_immediate_danger(test_state)
    if alert:
        print(f"✅ Alert generated: {alert.type}")
        print(f"   Message: {alert.message}")
        print(f"   Reaction time: {alert.reaction_time_ms:.1f}ms")
    else:
        print("❌ No alert (should have detected emergency)")

    # Test 2: Normal driving
    print("\nTest 2: Normal Driving")
    test_state = {
        "speed": 70,
        "leading_vehicle": {
            "distance": 50,
            "speed": 68,
            "acceleration": 0
        }
    }

    alert = reactive.check_immediate_danger(test_state)
    if alert:
        print(f"❌ Unexpected alert: {alert.type}")
    else:
        print("✅ Correctly no alert")

    # Test 3: Critical speed
    print("\nTest 3: Critical Speed")
    test_state = {
        "speed": 95,
        "speed_limit": 90
    }

    alert = reactive.check_immediate_danger(test_state)
    if alert and alert.type == "critical_speed":
        print(f"✅ Speed alert correctly detected")
    else:
        print("❌ Failed to detect speed violation")

    # Performance report
    print("\n📊 Performance Report:")
    report = reactive.get_performance_report()
    for key, value in report.items():
        if key != "alert_types":
            print(f"  {key}: {value}")

    print("\n✅ Reactive Layer Tests Complete!")


if __name__ == "__main__":
    # Run tests if this file is executed directly
    test_reactive_layer()