#!/usr/bin/env python3
"""
Quick integration test for the CCTV system
"""

def test_imports():
    """Test if all modules can be imported"""
    try:
        from main_simulation import main as run_main_simulation
        from report_generator import CCTVReportGenerator
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_main_simulation_structure():
    """Test if main simulation returns expected structure"""
    try:
        from main_simulation import CCTVSimulation

        # Quick simulation
        sim = CCTVSimulation(crime_probability=0.1)

        # Test training with 1 episode
        training_metrics = sim.train_q_learning(episodes=1)
        print("✅ Training function works")

        # Test evaluation structure
        ql_eval = sim.evaluate_q_learning()
        baseline_eval = sim.evaluate_baseline()

        required_keys = ['detection_rate', 'total_reward', 'total_crimes', 'detected_crimes']

        for key in required_keys:
            if key not in ql_eval:
                print(f"❌ Missing key in ql_eval: {key}")
                return False
            if key not in baseline_eval:
                print(f"❌ Missing key in baseline_eval: {key}")
                return False

        print("✅ Evaluation functions return correct structure")
        return True

    except Exception as e:
        print(f"❌ Main simulation test failed: {e}")
        return False

def test_report_generator():
    """Test if report generator can handle mock data"""
    try:
        from report_generator import CCTVReportGenerator

        # Mock results structure
        mock_results = {
            'training_metrics': {'q_table_size': 10, 'epsilon': 0.1},
            'q_learning_eval': {
                'detection_rate': 0.85,
                'total_reward': 1000,
                'total_crimes': 100,
                'detected_crimes': 85
            },
            'baseline_eval': {
                'detection_rate': 0.75,
                'total_reward': 750,
                'total_crimes': 100,
                'detected_crimes': 75
            },
            'visualization_data': [{'ql_action': 0, 'baseline_action': 1} for _ in range(10)],
            'training_rewards': [100, 150, 200],
            'training_detection_rates': [0.7, 0.8, 0.85],
            'epsilon_values': [1.0, 0.5, 0.1]
        }

        generator = CCTVReportGenerator(mock_results)
        print("✅ Report generator initialized successfully")
        return True

    except Exception as e:
        print(f"❌ Report generator test failed: {e}")
        return False

def main():
    print("🧪 Quick Integration Test")
    print("=" * 40)

    tests = [
        ("Import Test", test_imports),
        ("Main Simulation Structure", test_main_simulation_structure),
        ("Report Generator", test_report_generator)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\nRunning: {test_name}")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")

    print("\n" + "=" * 40)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Integration ready.")
        print("\nTo run full simulation:")
        print("  python integrated_simulation.py --quick      # 5-minute test")
        print("  python integrated_simulation.py --episodes 50 --no-video  # No video")
        print("  python integrated_simulation.py             # Full simulation")
    else:
        print("⚠️  Some tests failed. Check errors above.")

if __name__ == "__main__":
    main()