import time
from typing import Dict
from dataclasses import dataclass
from collections import deque

import core.agent.ReactiveLayer


@dataclass
class VehicleState:
    """Represents current vehicle state"""
    speed: float = 0.0
    position: tuple = (0.0, 0.0)
    lane_id: str = ""
    timestamp: float = 0.0
    leading_vehicle: Dict = None
    traffic_light_state: str = "green"


class HybridDrivingAgent:
    """Hybrid reactive + BDI agent for driving assistance"""

    def __init__(self, config: Dict):
        self.config = config
        self.vehicle_state = VehicleState()
        self.beliefs = {}
        self.desires = {"safety": 1.0, "information": 0.7, "comfort": 0.3}
        self.intentions = []
        self.alert_history = deque(maxlen=20)

        # Initialize components
        self.reactive_layer = ReactiveLayer(config)
        self.bdi_layer = BDILayer(config)
        self.nlg_engine = AlertGenerator(config)
        self.tts_engine = TTSEngine(config)

        # Performance tracking
        self.reaction_times = []
        self.last_update = time.time()

    def process_frame(self, frame_data: Dict) -> Dict:
        """
        Main processing loop for one frame of data
        Returns: Dict with decisions and alerts
        """
        start_time = time.time()

        # 1. Update vehicle state
        self._update_state(frame_data)

        # 2. Check reactive layer (CRITICAL path)
        reactive_alert = self.reactive_layer.check_immediate_danger(
            self.vehicle_state
        )

        if reactive_alert:
            # Bypass BDI for immediate response
            alert_msg = self.nlg_engine.generate_urgent_alert(reactive_alert)
            self.tts_engine.speak(alert_msg)
            return {
                "alert_type": "reactive",
                "message": alert_msg,
                "reaction_time": time.time() - start_time
            }

        # 3. Run BDI reasoning (if no immediate danger)
        context = self.bdi_layer.analyze_context(
            self.vehicle_state,
            self.beliefs
        )

        # 4. Generate appropriate alert if needed
        if context["should_alert"]:
            alert_msg = self.nlg_engine.generate_contextual_alert(
                context["situation"],
                self.vehicle_state
            )
            self.tts_engine.speak(alert_msg)

            return {
                "alert_type": "contextual",
                "message": alert_msg,
                "situation": context["situation"],
                "reaction_time": time.time() - start_time
            }

        return {"alert_type": "none", "reaction_time": time.time() - start_time}

    def _update_state(self, frame_data: Dict):
        """Update internal beliefs based on new data"""
        self.vehicle_state.speed = frame_data.get("speed", 0.0)
        self.vehicle_state.position = (
            frame_data.get("pos_x", 0.0),
            frame_data.get("pos_y", 0.0)
        )
        # ... update other state variables