#!/usr/bin/env python3
"""
CCTV Q-Learning Integrated Simulation System
============================================

This script runs the complete CCTV reinforcement learning simulation:
1. Trains Q-Learning agent for 365-day simulation
2. Creates 10-minute animation video
3. Generates comprehensive performance analysis
4. Produces detailed evaluation report

Usage:
    python integrated_simulation.py [--episodes N] [--no-video]
"""

import argparse
import sys
import time
from main_simulation import main as run_main_simulation
from report_generator import CCTVReportGenerator


def run_integrated_cctv_simulation(episodes=3000, eval_episodes=90, create_video=True):
    """
    Run the complete integrated CCTV simulation workflow

    Args:
        episodes (int): Number of training episodes (default: 1000)
        eval_episodes (int): Number of evaluation episodes (default: 90)
        create_video (bool): Whether to create animation video (default: True)

    Returns:
        dict: Complete simulation results and analysis
    """

    print("CCTV Q-Learning Integrated Simulation")
    print("=" * 60)
    print(f"Training Episodes: {episodes} (representing {episodes} days)")
    print(f"Evaluation Episodes: {eval_episodes} (representing {eval_episodes} days)")
    print(f"Video Generation: {'Enabled' if create_video else 'Disabled'}")
    print(f"Expected Duration: {'~20 minutes' if create_video else '~10 minutes'}")
    print("=" * 60)

    start_time = time.time()

    # Phase 1: Run Main Simulation (Training + Evaluation + Video)
    print("\nPHASE 1: Main Simulation")
    print("-" * 40)
    print(f"• Training Q-Learning agent ({episodes} episodes)")
    print(f"• Evaluating both systems ({eval_episodes} days)")
    if create_video:
        print("• Creating 10-minute animation video")

    try:
        main_results = run_main_simulation(episodes=episodes, eval_episodes=eval_episodes, create_video=create_video)
    except Exception as e:
        print(f"❌ Error in main simulation: {e}")
        return None

    phase1_time = time.time() - start_time
    print(f"✅ Phase 1 completed in {phase1_time:.1f} seconds")

    # Phase 2: Generate Comprehensive Report
    print("\nPHASE 2: Comprehensive Analysis")
    print("-" * 40)
    print("• Creating 9-panel performance analysis")
    print("• Generating intersection visualization")
    print("• Writing detailed evaluation report")

    try:
        report_generator = CCTVReportGenerator(main_results)
        analysis_results = report_generator.run_comprehensive_analysis()
    except Exception as e:
        print(f"Error in report generation: {e}")
        return None

    total_time = time.time() - start_time
    print(f"Phase 2 completed in {total_time - phase1_time:.1f} seconds")

    # Summary
    print("\nSIMULATION COMPLETE!")
    print("=" * 60)
    print(f"Total Duration: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print("\nGenerated Files:")
    print("   • cctv_simulation.mp4 - 10-minute animation video" if create_video else "   • (Video generation skipped)")
    print("   • comprehensive_performance_analysis.png - 9-panel analysis")
    print("   • intersection_comparison.png - System comparison")
    print("   • evaluation_report.txt - Detailed report")
    print("   • training_results.png - Training metrics")

    # Performance Summary
    ql_rate = main_results['q_learning_eval']['detection_rate']
    baseline_rate = main_results['baseline_eval']['detection_rate']

    print(f"\nPerformance Summary ({eval_episodes}-day evaluation):")
    print(f"   • Q-Learning Detection Rate: {ql_rate:.3f}")
    print(f"   • Sequential Detection Rate: {baseline_rate:.3f}")

    if baseline_rate > 0:
        improvement = ((ql_rate - baseline_rate) / baseline_rate) * 100
        print(f"   • Performance Improvement: {improvement:+.1f}%")

    print("=" * 60)

    return {
        'main_results': main_results,
        'analysis_results': analysis_results,
        'execution_time': total_time,
        'files_generated': [
            'cctv_simulation.mp4' if create_video else None,
            'comprehensive_performance_analysis.png',
            'intersection_comparison.png',
            'evaluation_report.txt'
        ]
    }


def main():
    """Command line interface for the integrated simulation"""
    parser = argparse.ArgumentParser(
        description='CCTV Q-Learning Integrated Simulation System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python integrated_simulation.py                    # Default: 1000 training episodes, 90 eval episodes with video
  python integrated_simulation.py --episodes 500     # Run with 500 training episodes
  python integrated_simulation.py --no-video         # Skip video generation
  python integrated_simulation.py --quick            # Quick test: 100 training, 10 eval, no video
  python integrated_simulation.py --episodes 2000 --eval-episodes 180  # Extended run
        """
    )

    parser.add_argument(
        '--episodes',
        type=int,
        default=3000,
        help='Number of training episodes (default: 3000)'
    )

    parser.add_argument(
        '--eval-episodes',
        type=int,
        default=90,
        help='Number of evaluation episodes (default: 90)'
    )

    parser.add_argument(
        '--no-video',
        action='store_true',
        help='Skip video generation to save time'
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test run (100 episodes, 10 eval episodes, no video)'
    )

    args = parser.parse_args()

    # Handle quick mode
    if args.quick:
        episodes = 100
        eval_episodes = 10
        create_video = False
        print("Quick Mode: 100 training episodes, 10 eval episodes, no video")
    else:
        episodes = args.episodes
        eval_episodes = args.eval_episodes
        create_video = not args.no_video

    # Validation
    if episodes < 1:
        print("❌ Error: Episodes must be at least 1")
        sys.exit(1)

    if episodes > 5000:
        print("Warning: Large number of episodes may take a very long time")
        response = input("Continue? (y/N): ")
        if response.lower() != 'y':
            print("Simulation cancelled")
            sys.exit(0)

    # Run simulation
    try:
        results = run_integrated_cctv_simulation(episodes=episodes, eval_episodes=eval_episodes, create_video=create_video)

        if results is None:
            print("Simulation failed")
            sys.exit(1)
        else:
            print(f"\nAll files saved in: /home/leehs8006/tch/cctv_control/")
            sys.exit(0)

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()