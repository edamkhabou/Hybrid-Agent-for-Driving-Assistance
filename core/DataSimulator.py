import random
import time
from typing import Dict


class DrivingScenarioSimulator:
    """Generates mock driving data for development"""

    def __init__(self):
        self.scenarios = {
            "normal": self._normal_driving,
            "overtaking": self._overtaking_scenario,
            "intersection": self._intersection_scenario,
            "danger_zone": self._danger_zone_scenario
        }
        self.current_scenario = "normal"
        self.scenario_progress = 0

    def generate_frame(self) -> Dict:
        """Generate one frame of simulated vehicle data"""
        frame = self.scenarios[self.current_scenario]()

        # Occasionally change scenario
        if random.random() < 0.01:  # 1% chance to change
            self.current_scenario = random.choice(list(self.scenarios.keys()))
            self.scenario_progress = 0

        self.scenario_progress += 1
        frame["timestamp"] = time.time()
        frame["scenario"] = self.current_scenario

        return frame

    def _normal_driving(self) -> Dict:
        return {
            "speed": 50 + random.uniform(-5, 5),
            "pos_x": 1000 + self.scenario_progress * 10,
            "pos_y": 250,
            "lane_id": "highway_lane_1",
            "leading_vehicle": {
                "distance": 30 + random.uniform(-5, 5),
                "speed": 48 + random.uniform(-3, 3)
            },
            "traffic_light_state": "green"
        }

    def _overtaking_scenario(self) -> Dict:
        # Simulate vehicle approaching slower vehicle
        base_speed = 60
        if self.scenario_progress < 20:
            # Approaching
            distance = max(50 - self.scenario_progress * 2, 5)
        elif self.scenario_progress < 40:
            # Overtaking
            distance = 5
            base_speed = 65
        else:
            # After overtaking
            distance = 10 + self.scenario_progress * 0.5

        return {
            "speed": base_speed + random.uniform(-2, 2),
            "pos_x": 1000 + self.scenario_progress * 12,
            "pos_y": 250,
            "lane_id": "highway_lane_1",
            "leading_vehicle": {
                "distance": distance,
                "speed": 55 if self.scenario_progress < 30 else 58
            },
            "left_lane_vehicle": {
                "distance": 15 + random.uniform(-3, 3),
                "relative_speed": 5 if self.scenario_progress < 30 else -2
            }
        }

    def _intersection_scenario(self) -> Dict:
        # Simulate approaching intersection
        distance_to_intersection = max(200 - self.scenario_progress * 8, 0)

        return {
            "speed": 40 + random.uniform(-3, 3),
            "pos_x": 500 + self.scenario_progress * 7,
            "pos_y": 300,
            "lane_id": "urban_lane_2",
            "next_intersection": {
                "distance": distance_to_intersection,
                "light_state": "red" if self.scenario_progress > 15 else "green"
            },
            "cross_traffic": random.random() > 0.7
        }