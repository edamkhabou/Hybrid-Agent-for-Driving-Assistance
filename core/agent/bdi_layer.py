"""
BDI Layer: Belief-Desire-Intention reasoning for contextual decisions
"""
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
import numpy as np

from config.settings import AGENT_CONFIG, AgentConfig


@dataclass
class Belief:
    """Agent's beliefs about the current driving context"""
    timestamp: float
    vehicle_state: Dict[str, Any]
    context_probs: Dict[str, float]  # Context probabilities from ML
    surroundings: Dict[str, Any]
    road_conditions: Dict[str, Any]

    def get_primary_context(self, threshold: float = 0.7) -> str:
        """Get the primary driving context based on probabilities"""
        if not self.context_probs:
            return "unknown"

        sorted_contexts = sorted(
            self.context_probs.items(),
            key=lambda x: x[1],
            reverse=True
        )

        primary, confidence = sorted_contexts[0]
        return primary if confidence > threshold else "uncertain"

    def get_context_confidence(self, context: str) -> float:
        """Get confidence level for a specific context"""
        return self.context_probs.get(context, 0.0)


@dataclass
class Desire:
    """Agent's desires with weights"""
    safety: float = 0.0  # Desire for safety (0-1)
    information: float = 0.0  # Desire for information
    comfort: float = 0.0  # Desire for comfort
    efficiency: float = 0.0  # Desire for efficiency

    def weighted_sum(self, weights: Dict[str, float]) -> float:
        """Calculate weighted sum of desires"""
        return (
                self.safety * weights.get("safety", 0) +
                self.information * weights.get("information", 0) +
                self.comfort * weights.get("comfort", 0) +
                self.efficiency * weights.get("efficiency", 0)
        )


@dataclass
class Intention:
    """Agent's intention/planned action"""
    action_type: str
    utility: float  # Expected utility (0-1)
    parameters: Dict[str, Any]
    priority: int  # 1=highest, 3=lowest
    requires_alert: bool = True

    def __lt__(self, other):
        # Sort by priority first, then utility
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.utility > other.utility


class BDILayer:
    """
    Belief-Desire-Intention reasoning layer.
    Handles contextual analysis and strategic decision making.
    """

    def __init__(self, config: AgentConfig = None):
        self.config = config or AGENT_CONFIG
        self.beliefs_history = deque(maxlen=50)  # Keep last 50 beliefs
        self.intentions_history = deque(maxlen=20)
        self.current_belief: Optional[Belief] = None
        self.current_desire: Optional[Desire] = None
        self.current_intentions: List[Intention] = []

        self.performance_stats = {
            "total_cycles": 0,
            "avg_cycle_time_ms": 0.0,
            "max_cycle_time_ms": 0.0,
            "intentions_generated": 0
        }

        print("✅ BDI Layer initialized")

    def process_context(self,
                        vehicle_state: Dict,
                        ml_predictions: Dict) -> List[Intention]:
        """
        Main BDI processing cycle
        Returns: List of intentions sorted by priority and utility
        """
        start_time = time.perf_counter()
        self.performance_stats["total_cycles"] += 1

        # 1. UPDATE BELIEFS
        self.current_belief = self._update_beliefs(vehicle_state, ml_predictions)

        # 2. EVALUATE DESIRES
        self.current_desire = self._evaluate_desires(self.current_belief)

        # 3. FORM INTENTIONS
        self.current_intentions = self._form_intentions(
            self.current_belief,
            self.current_desire
        )

        # 4. SELECT INTENTIONS (filter by utility threshold)
        selected_intentions = [
            intention for intention in self.current_intentions
            if intention.utility >= self.config.UTILITY_THRESHOLD
        ]

        # Sort by priority and utility
        selected_intentions.sort()

        # Update history
        self.intentions_history.extend(selected_intentions)
        self.performance_stats["intentions_generated"] += len(selected_intentions)

        # Calculate cycle time
        cycle_time = (time.perf_counter() - start_time) * 1000
        self._update_performance_stats(cycle_time)

        return selected_intentions

    def _update_beliefs(self,
                        vehicle_state: Dict,
                        ml_predictions: Dict) -> Belief:
        """Update agent's beliefs based on new observations"""
        surroundings = self._analyze_surroundings(vehicle_state)
        road_conditions = self._analyze_road_conditions(vehicle_state)

        belief = Belief(
            timestamp=time.time(),
            vehicle_state=vehicle_state,
            context_probs=ml_predictions,
            surroundings=surroundings,
            road_conditions=road_conditions
        )

        self.beliefs_history.append(belief)
        return belief

    def _evaluate_desires(self, belief: Belief) -> Desire:
        """Evaluate desires based on current beliefs"""
        desire = Desire()

        # Safety desire (based on risk factors)
        desire.safety = self._calculate_safety_desire(belief)

        # Information desire (based on uncertainty)
        desire.information = self._calculate_information_desire(belief)

        # Comfort desire (based on driving smoothness)
        desire.comfort = self._calculate_comfort_desire(belief)

        # Efficiency desire (based on progress toward destination)
        desire.efficiency = self._calculate_efficiency_desire(belief)

        return desire

    def _form_intentions(self, belief: Belief, desire: Desire) -> List[Intention]:
        """Form intentions based on beliefs and desires"""
        intentions = []

        # Get possible actions based on context
        possible_actions = self._generate_possible_actions(belief)

        for action in possible_actions:
            utility = self._calculate_action_utility(action, belief, desire)
            priority = self._determine_action_priority(action["type"])

            intention = Intention(
                action_type=action["type"],
                utility=utility,
                parameters=action.get("parameters", {}),
                priority=priority,
                requires_alert=action.get("requires_alert", True)
            )

            intentions.append(intention)

        return intentions

    def _analyze_surroundings(self, state: Dict) -> Dict:
        """Analyze vehicle surroundings"""
        surroundings = {
            "traffic_density": self._calculate_traffic_density(state),
            "lane_position": self._determine_lane_position(state),
            "intersection_proximity": state.get("distance_to_intersection", 1000),
            "overtaking_opportunity": self._check_overtaking_opportunity(state),
            "blind_spots": self._check_blind_spots(state)
        }
        return surroundings

    def _analyze_road_conditions(self, state: Dict) -> Dict:
        """Analyze current road conditions"""
        return {
            "road_type": state.get("road_type", "unknown"),
            "visibility": state.get("visibility", "good"),
            "surface_condition": state.get("surface_condition", "dry"),
            "curvature": state.get("road_curvature", 0)
        }

    def _calculate_safety_desire(self, belief: Belief) -> float:
        """Calculate safety desire (higher when risk is high)"""
        risk_score = 0.0

        # Risk from speed
        speed = belief.vehicle_state.get("speed", 0)
        speed_limit = belief.vehicle_state.get("speed_limit", 90)
        if speed > speed_limit:
            risk_score += 0.3

        # Risk from following distance
        if "leading_vehicle" in belief.vehicle_state:
            lead = belief.vehicle_state["leading_vehicle"]
            distance = lead.get("distance", 1000)
            ego_speed = speed / 3.6  # m/s

            # 2-second rule violation
            if distance < ego_speed * 2:
                risk_score += 0.4

        # Risk from context
        context = belief.get_primary_context()
        if context in ["danger_zone", "emergency"]:
            risk_score += 0.3

        return min(1.0, risk_score)

    def _calculate_information_desire(self, belief: Belief) -> float:
        """Calculate information desire (higher when uncertain)"""
        # High uncertainty in context
        if belief.get_primary_context() == "uncertain":
            return 0.8

        # Complex situation (intersection, merging, etc.)
        context_probs = belief.context_probs
        complex_contexts = ["intersection", "merging", "lane_change"]
        complex_score = sum(context_probs.get(ctx, 0) for ctx in complex_contexts)

        return min(0.7, complex_score * 0.8)

    def _calculate_comfort_desire(self, belief: Belief) -> float:
        """Calculate comfort desire"""
        # Lower comfort during aggressive maneuvers
        acceleration = abs(belief.vehicle_state.get("acceleration", 0))
        if acceleration > 3.0:
            return 0.2
        elif acceleration > 2.0:
            return 0.4
        return 0.8

    def _calculate_efficiency_desire(self, belief: Belief) -> float:
        """Calculate efficiency desire"""
        # Higher when maintaining optimal speed
        speed = belief.vehicle_state.get("speed", 0)
        optimal_speed = 80  # km/h
        speed_diff = abs(speed - optimal_speed)

        if speed_diff < 10:
            return 0.9
        elif speed_diff < 20:
            return 0.6
        return 0.3

    def _generate_possible_actions(self, belief: Belief) -> List[Dict]:
        """Generate possible actions based on context"""
        actions = []
        primary_context = belief.get_primary_context()

        # Context-specific actions
        if primary_context == "overtaking":
            actions.append({
                "type": "alert_overtaking_opportunity",
                "parameters": {
                    "confidence": belief.get_context_confidence("overtaking"),
                    "recommended_action": "Check mirrors and blind spot before overtaking"
                },
                "requires_alert": True
            })

        elif primary_context == "intersection":
            distance = belief.surroundings.get("intersection_proximity", 1000)
            actions.append({
                "type": "alert_intersection_approach",
                "parameters": {
                    "distance": distance,
                    "light_state": belief.vehicle_state.get("traffic_light_state", "unknown"),
                    "recommended_action": "Prepare to stop" if distance < 100 else "Monitor traffic"
                },
                "requires_alert": True
            })

        elif primary_context == "danger_zone":
            actions.append({
                "type": "alert_danger_zone",
                "parameters": {
                    "severity": "high",
                    "recommended_action": "Increase following distance and reduce speed"
                },
                "requires_alert": True
            })

        # Always available informational actions
        actions.extend([
            {
                "type": "provide_speed_feedback",
                "parameters": {
                    "current_speed": belief.vehicle_state.get("speed", 0),
                    "speed_limit": belief.vehicle_state.get("speed_limit", 90),
                    "efficiency": "optimal" if belief.vehicle_state.get("speed", 0) > 75 else "suboptimal"
                },
                "requires_alert": False
            },
            {
                "type": "update_navigation_guidance",
                "parameters": {
                    "lane_recommendation": belief.surroundings.get("lane_position", "keep"),
                    "next_maneuver": "continue"  # Could be "turn", "exit", etc.
                },
                "requires_alert": False
            }
        ])

        return actions

    def _calculate_action_utility(self,
                                  action: Dict,
                                  belief: Belief,
                                  desire: Desire) -> float:
        """Calculate utility of an action"""
        base_utility = 0.0
        action_type = action["type"]

        if action_type.startswith("alert_"):
            # Alert actions primarily satisfy safety and information desires
            base_utility = (
                    desire.safety * 0.6 +
                    desire.information * 0.4
            )

            # Boost utility for high-risk contexts
            if belief.get_primary_context() in ["danger_zone", "emergency"]:
                base_utility *= 1.3

        elif "speed_feedback" in action_type:
            base_utility = desire.information * 0.7 + desire.efficiency * 0.3

        elif "navigation" in action_type:
            base_utility = desire.efficiency * 0.8 + desire.comfort * 0.2

        return min(1.0, base_utility)

    def _determine_action_priority(self, action_type: str) -> int:
        """Determine priority of an action (1=highest, 3=lowest)"""
        priority_map = {
            "alert_danger_zone": 1,
            "alert_intersection_approach": 1,
            "alert_overtaking_opportunity": 2,
            "provide_speed_feedback": 3,
            "update_navigation_guidance": 3
        }
        return priority_map.get(action_type, 3)

    # Helper methods
    def _calculate_traffic_density(self, state: Dict) -> float:
        """Calculate traffic density (0-1)"""
        nearby = state.get("nearby_vehicles", [])
        return min(1.0, len(nearby) / 10.0)

    def _determine_lane_position(self, state: Dict) -> str:
        """Determine lane position from lane ID"""
        lane = state.get("lane", "")
        if "left" in lane.lower():
            return "left_lane"
        elif "right" in lane.lower():
            return "right_lane"
        return "center_lane"

    def _check_overtaking_opportunity(self, state: Dict) -> bool:
        """Check if overtaking is currently possible"""
        return (
                state.get("left_lane_free", False) and
                state.get("distance_to_lead", 1000) < 50 and
                state.get("speed", 0) > state.get("leading_vehicle", {}).get("speed", 0) + 10
        )

    def _check_blind_spots(self, state: Dict) -> List[str]:
        """Check which blind spots have vehicles"""
        blind_spots = []
        nearby = state.get("nearby_vehicles", [])

        for vehicle in nearby:
            rel_position = vehicle.get("relative_position", "")
            if "blind_spot" in rel_position.lower():
                blind_spots.append(rel_position)

        return blind_spots

    def _update_performance_stats(self, cycle_time: float):
        """Update performance statistics"""
        self.performance_stats["avg_cycle_time_ms"] = (
                self.performance_stats["avg_cycle_time_ms"] * 0.9 + cycle_time * 0.1
        )
        self.performance_stats["max_cycle_time_ms"] = max(
            self.performance_stats["max_cycle_time_ms"], cycle_time
        )

        # Warn if too slow
        if cycle_time > self.config.MAX_BDI_CYCLE_MS:
            print(f"⚠️  Warning: BDI cycle took {cycle_time:.1f}ms (> {self.config.MAX_BDI_CYCLE_MS}ms)")

    def get_current_state(self) -> Dict:
        """Get current BDI state for debugging"""
        return {
            "belief": asdict(self.current_belief) if self.current_belief else None,
            "desire": asdict(self.current_desire) if self.current_desire else None,
            "intentions": [asdict(i) for i in self.current_intentions],
            "performance": self.performance_stats
        }

    def get_performance_report(self) -> Dict:
        """Get performance report"""
        return {
            **self.performance_stats,
            "meets_sla": self.performance_stats.get("max_cycle_time_ms", 0) < self.config.MAX_BDI_CYCLE_MS,
            "avg_desire_safety": np.mean([d.safety for d in self.beliefs_history]) if self.beliefs_history else 0,
            "recent_intentions": [
                {"type": i.action_type, "utility": i.utility}
                for i in list(self.intentions_history)[-5:]
            ]
        }


# Test function for PyCharm
def test_bdi_layer():
    """Test the BDI layer implementation"""
    print("\n🧪 Testing BDI Layer...")

    bdi = BDILayer()

    # Test 1: Overtaking context
    print("\nTest 1: Overtaking Scenario")
    vehicle_state = {
        "speed": 90,
        "speed_limit": 110,
        "lane": "highway_left_0",
        "leading_vehicle": {"distance": 30, "speed": 80},
        "left_lane_free": True,
        "nearby_vehicles": [
            {"id": "veh1", "relative_position": "left_rear"}
        ]
    }

    ml_predictions = {
        "overtaking": 0.85,
        "normal": 0.10,
        "intersection": 0.03,
        "danger_zone": 0.02
    }

    intentions = bdi.process_context(vehicle_state, ml_predictions)

    print(f"✅ Generated {len(intentions)} intentions:")
    for i, intention in enumerate(intentions, 1):
        print(f"  {i}. {intention.action_type} (utility: {intention.utility:.2f}, priority: {intention.priority})")

    # Test 2: Intersection context
    print("\nTest 2: Intersection Scenario")
    vehicle_state = {
        "speed": 50,
        "speed_limit": 50,
        "lane": "urban_approach_0",
        "distance_to_intersection": 80,
        "traffic_light_state": "yellow"
    }

    ml_predictions = {
        "intersection": 0.92,
        "normal": 0.05,
        "danger_zone": 0.02,
        "overtaking": 0.01
    }

    intentions = bdi.process_context(vehicle_state, ml_predictions)

    print(f"✅ Generated {len(intentions)} intentions:")
    for i, intention in enumerate(intentions, 1):
        print(f"  {i}. {intention.action_type} (utility: {intention.utility:.2f})")

    # Performance report
    print("\n📊 BDI Performance Report:")
    report = bdi.get_performance_report()
    for key, value in report.items():
        if key not in ["recent_intentions", "avg_desire_safety"]:
            print(f"  {key}: {value}")

    print("\n✅ BDI Layer Tests Complete!")


if __name__ == "__main__":
    test_bdi_layer()