import time

from core.DataSimulator import DrivingScenarioSimulator
from core.HybridDrivingAgent import HybridDrivingAgent


def main():
    print("🚗 Starting Intelligent Driving Assistant Agent")

    # Load configuration
    config = {
        "critical_ttc": 1.5,
        "critical_speed": 90,
        "min_safe_distance": 3.0,
        "alert_cooldown": 2.0  # seconds between alerts
    }

    # Initialize agent
    agent = HybridDrivingAgent(config)

    # Initialize simulator (replace with VEINS interface later)
    simulator = DrivingScenarioSimulator()

    print("✅ Agent initialized. Starting simulation loop...")
    print("-" * 50)

    # Main simulation loop
    try:
        for frame_count in range(1000):  # Simulate 1000 frames
            # Get simulated data (replace with real VEINS data)
            frame_data = simulator.generate_frame()

            # Process with agent
            result = agent.process_frame(frame_data)

            # Display results
            if result["alert_type"] != "none":
                print(f"\nFrame {frame_count}:")
                print(f"  Scenario: {frame_data.get('scenario', 'unknown')}")
                print(f"  Speed: {frame_data.get('speed', 0):.1f} km/h")
                print(f"  Alert: {result['alert_type']}")
                print(f"  Message: {result.get('message', '')}")
                print(f"  Reaction time: {result['reaction_time'] * 1000:.1f} ms")
                print("-" * 30)

            # Simulate real-time pacing (e.g., 10 Hz)
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n🛑 Simulation stopped by user")

    # Print performance summary
    print("\n📊 Performance Summary:")
    print(f"Total frames processed: {frame_count}")
    if agent.reaction_times:
        avg_time = sum(agent.reaction_times) / len(agent.reaction_times) * 1000
        print(f"Average reaction time: {avg_time:.1f} ms")


if __name__ == "__main__":
    main()