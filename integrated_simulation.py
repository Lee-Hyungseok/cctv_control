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


def run_integrated_cctv_simulation(episodes=100, create_video=True):
    """
    Run the complete integrated CCTV simulation workflow

    Args:
        episodes (int): Number of training episodes (default: 100)
        create_video (bool): Whether to create animation video (default: True)

    Returns:
        dict: Complete simulation results and analysis
    """

    print("🎬 CCTV Q-Learning Integrated Simulation")
    print("=" * 60)
    print(f"Training Episodes: {episodes}")
    print(f"Video Generation: {'Enabled' if create_video else 'Disabled'}")
    print(f"Expected Duration: {'~15 minutes' if create_video else '~10 minutes'}")
    print("=" * 60)

    start_time = time.time()

    # Phase 1: Run Main Simulation (Training + Evaluation + Video)
    print("\n🚀 PHASE 1: Main Simulation")
    print("-" * 40)
    print("• Training Q-Learning agent")
    print("• Evaluating both systems (365 days)")
    if create_video:
        print("• Creating 10-minute animation video")

    try:
        main_results = run_main_simulation(episodes=episodes, create_video=create_video)
    except Exception as e:
        print(f"❌ Error in main simulation: {e}")
        return None

    phase1_time = time.time() - start_time
    print(f"✅ Phase 1 completed in {phase1_time:.1f} seconds")

    # Phase 2: Generate Comprehensive Report
    print("\n📊 PHASE 2: Comprehensive Analysis")
    print("-" * 40)
    print("• Creating 9-panel performance analysis")
    print("• Generating intersection visualization")
    print("• Writing detailed evaluation report")

    try:
        report_generator = CCTVReportGenerator(main_results)
        analysis_results = report_generator.run_comprehensive_analysis()
    except Exception as e:
        print(f"❌ Error in report generation: {e}")
        return None

    total_time = time.time() - start_time
    print(f"✅ Phase 2 completed in {total_time - phase1_time:.1f} seconds")

    # Summary
    print("\n🎉 SIMULATION COMPLETE!")
    print("=" * 60)
    print(f"Total Duration: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print("\n📁 Generated Files:")
    print("   • cctv_simulation.mp4 - 10-minute animation video" if create_video else "   • (Video generation skipped)")
    print("   • comprehensive_performance_analysis.png - 9-panel analysis")
    print("   • intersection_comparison.png - System comparison")
    print("   • evaluation_report.txt - Detailed report")

    # Performance Summary
    ql_rate = main_results['q_learning_eval']['detection_rate']
    baseline_rate = main_results['baseline_eval']['detection_rate']

    print(f"\n🎯 Performance Summary:")
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
  python integrated_simulation.py                    # Default: 100 episodes with video
  python integrated_simulation.py --episodes 50     # Quick run with 50 episodes
  python integrated_simulation.py --no-video        # Skip video generation
  python integrated_simulation.py --episodes 200 --no-video  # Longer training, no video
        """
    )

    parser.add_argument(
        '--episodes',
        type=int,
        default=100,
        help='Number of training episodes (default: 100)'
    )

    parser.add_argument(
        '--no-video',
        action='store_true',
        help='Skip video generation to save time'
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test run (50 episodes, no video)'
    )

    args = parser.parse_args()

    # Handle quick mode
    if args.quick:
        episodes = 50
        create_video = False
        print("🚀 Quick Mode: 50 episodes, no video")
    else:
        episodes = args.episodes
        create_video = not args.no_video

    # Validation
    if episodes < 1:
        print("❌ Error: Episodes must be at least 1")
        sys.exit(1)

    if episodes > 1000:
        print("⚠️  Warning: Large number of episodes may take a very long time")
        response = input("Continue? (y/N): ")
        if response.lower() != 'y':
            print("Simulation cancelled")
            sys.exit(0)

    # Run simulation
    try:
        results = run_integrated_cctv_simulation(episodes=episodes, create_video=create_video)

        if results is None:
            print("❌ Simulation failed")
            sys.exit(1)
        else:
            print(f"\n✅ All files saved in: /home/leehs8006/tch/cctv_control/")
            sys.exit(0)

    except KeyboardInterrupt:
        print("\n⚠️  Simulation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()