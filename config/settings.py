"""
Configuration for the agent layers
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class AgentConfig:
    # Reactive Layer Settings
    CRITICAL_TTC: float = 1.5  # seconds for emergency braking
    WARNING_TTC: float = 3.0  # seconds for warning
    CRITICAL_SPEED: float = 90.0  # km/h
    MIN_SAFE_DISTANCE: float = 3.0  # meters
    EMERGENCY_DECEL_THRESHOLD: float = 7.5  # m/s²

    # BDI Layer Settings
    DESIRE_WEIGHTS: Dict[str, float] = None
    ALERT_COOLDOWN: float = 2.0  # seconds between alerts
    UTILITY_THRESHOLD: float = 0.7

    # Performance
    MAX_REACTIVE_TIME_MS: float = 50.0  # 50ms max for reactive
    MAX_BDI_CYCLE_MS: float = 100.0  # 100ms max for BDI

    def __post_init__(self):
        if self.DESIRE_WEIGHTS is None:
            self.DESIRE_WEIGHTS = {
                "safety": 1.0,
                "information": 0.7,
                "comfort": 0.4,
                "efficiency": 0.3
            }


@dataclass
class VehicleConfig:
    EGO_VEHICLE_ID: str = "ego_vehicle"
    VEHICLE_LENGTH: float = 4.5  # meters
    MAX_DECELERATION: float = 9.0  # m/s²
    MAX_ACCELERATION: float = 3.0  # m/s²
    REACTION_TIME: float = 1.5  # seconds (human reaction)


# Global configurations
AGENT_CONFIG = AgentConfig()
VEHICLE_CONFIG = VehicleConfig()