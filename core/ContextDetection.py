import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib


class ContextDetector:
    """ML model for detecting driving contexts"""

    def __init__(self, model_path=None):
        if model_path:
            self.load_model(model_path)
        else:
            # Initialize with a simple model
            self.model = RandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                random_state=42
            )
            self.scaler = StandardScaler()
            self.is_trained = False

    def extract_features(self, vehicle_state: Dict) -> np.ndarray:
        """Extract features from vehicle state for ML model"""
        features = []

        # Basic features
        features.append(vehicle_state.get("speed", 0))
        features.append(vehicle_state.get("acceleration", 0))

        # Relative features
        if vehicle_state.get("leading_vehicle"):
            features.append(vehicle_state["leading_vehicle"].get("distance", 100))
            features.append(vehicle_state["leading_vehicle"].get("relative_speed", 0))

        # Lane features
        features.append(1 if "highway" in vehicle_state.get("lane_id", "") else 0)
        features.append(1 if "urban" in vehicle_state.get("lane_id", "") else 0)

        # Context indicators
        features.append(vehicle_state.get("distance_to_intersection", 1000))
        features.append(1 if vehicle_state.get("traffic_light_state") == "red" else 0)

        return np.array(features).reshape(1, -1)

    def predict_context(self, vehicle_state: Dict) -> Dict:
        """Predict driving context from vehicle state"""
        if not self.is_trained:
            # Return dummy probabilities for development
            return self._dummy_predictions(vehicle_state)

        # Extract features
        features = self.extract_features(vehicle_state)
        features_scaled = self.scaler.transform(features)

        # Get predictions
        probabilities = self.model.predict_proba(features_scaled)[0]

        return {
            "overtaking": probabilities[0],
            "intersection": probabilities[1],
            "danger_zone": probabilities[2],
            "normal": probabilities[3] if len(probabilities) > 3 else 0
        }

    def _dummy_predictions(self, vehicle_state: Dict) -> Dict:
        """Dummy predictions for development before model is trained"""
        # Simple rule-based "model" for prototyping
        speed = vehicle_state.get("speed", 0)

        # Simulate different contexts based on speed and other factors
        if speed > 70:
            return {"overtaking": 0.7, "intersection": 0.1, "danger_zone": 0.1, "normal": 0.1}
        elif vehicle_state.get("distance_to_intersection", 1000) < 100:
            return {"overtaking": 0.1, "intersection": 0.8, "danger_zone": 0.05, "normal": 0.05}
        else:
            return {"overtaking": 0.1, "intersection": 0.1, "danger_zone": 0.1, "normal": 0.7}