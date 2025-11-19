import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from typing import Dict

class CCTVReportGenerator:
    def __init__(self, main_results: Dict):
        self.results = main_results
        self.simulation = main_results['simulation']
        self.training_results = main_results['training_metrics']
        self.ql_eval = main_results['q_learning_eval']
        self.baseline_eval = main_results['baseline_eval']
        self.viz_data = main_results['visualization_data']

        # Extract direction-wise statistics from simulation object
        self.eval_stats = self.simulation.eval_stats
        self.training_stats = self.simulation.training_stats

    def create_comprehensive_performance_plots(self):
        """Create detailed performance analysis with 9 subplots"""
        fig = plt.figure(figsize=(20, 16))

        # Plot 1: Training Rewards (Cumulative)
        plt.subplot(3, 3, 1)
        training_rewards = self.results['training_rewards']
        cumulative_rewards = np.cumsum(training_rewards)
        plt.plot(cumulative_rewards, 'b-', linewidth=2)
        plt.title('Cumulative Training Rewards', fontsize=14, fontweight='bold')
        plt.xlabel('Episode')
        plt.ylabel('Cumulative Reward')
        plt.grid(True, alpha=0.3)

        # Plot 2: Cumulative Detection Rate During Training
        plt.subplot(3, 3, 2)
        # Calculate cumulative detection rate from training stats
        total_detections = self.training_stats['q_learning']['total_detections']
        total_crimes = self.training_stats['q_learning']['total_crimes']
        cumulative_detection_rate = total_detections / max(1, total_crimes)

        # For visualization, show moving average of detection rates
        detection_rates = self.results['training_detection_rates']
        window = min(50, len(detection_rates) // 10)
        if window > 1:
            smoothed_rates = [np.mean(detection_rates[max(0, i-window):i+1])
                             for i in range(len(detection_rates))]
            plt.plot(smoothed_rates, 'g-', linewidth=2)
            plt.title(f'Training Detection Rate (MA-{window})', fontsize=14, fontweight='bold')
        else:
            plt.plot(detection_rates, 'g-', linewidth=2)
            plt.title('Training Detection Rate', fontsize=14, fontweight='bold')

        plt.axhline(y=cumulative_detection_rate, color='r', linestyle='--',
                   label=f'Overall: {cumulative_detection_rate:.3f}', linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Detection Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 3: Epsilon Decay
        plt.subplot(3, 3, 3)
        epsilon_values = self.results['epsilon_values']
        plt.plot(epsilon_values, 'r-', linewidth=2)
        plt.title('Epsilon Decay During Training', fontsize=14, fontweight='bold')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon Value')
        plt.grid(True, alpha=0.3)

        # Plot 4: Detection Rate Comparison
        plt.subplot(3, 3, 4)
        methods = ['Sequential\nCCTV', 'Q-Learning\nCCTV']
        detection_rates_comp = [
            self.baseline_eval['detection_rate'],
            self.ql_eval['detection_rate']
        ]
        colors = ['#ff7f7f', '#7f7fff']
        bars = plt.bar(methods, detection_rates_comp, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        plt.title('Detection Rate Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('Detection Rate')
        plt.ylim(0, max(detection_rates_comp) * 1.1)

        for bar, rate in zip(bars, detection_rates_comp):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{rate:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)

        # Plot 5: Total Rewards Comparison
        plt.subplot(3, 3, 5)
        total_rewards = [
            self.baseline_eval['total_reward'],
            self.ql_eval['total_reward']
        ]
        bars = plt.bar(methods, total_rewards, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        plt.title('Total Rewards Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('Total Reward (365 days)')

        for bar, reward in zip(bars, total_rewards):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(total_rewards)*0.01,
                    f'{reward:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=12)

        # Plot 6: Crimes Detected Comparison
        plt.subplot(3, 3, 6)
        detected_crimes = [
            self.baseline_eval['detected_crimes'],
            self.ql_eval['detected_crimes']
        ]
        bars = plt.bar(methods, detected_crimes, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        plt.title('Crimes Detected Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('Number of Crimes Detected')

        for bar, crimes in zip(bars, detected_crimes):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(detected_crimes)*0.01,
                    f'{crimes:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=12)

        # Plot 7: Direction-wise Detection Rates (Q-Learning Evaluation)
        plt.subplot(3, 3, 7)
        directions = ['NORTH', 'SOUTH', 'EAST', 'WEST']
        ql_rates = []
        for direction in directions:
            crimes = self.eval_stats['q_learning']['crimes_by_direction'][direction]
            detections = self.eval_stats['q_learning']['detections_by_direction'][direction]
            rate = detections / max(1, crimes)
            ql_rates.append(rate)

        x_pos = np.arange(len(directions))
        bars = plt.bar(x_pos, ql_rates, color=['#ff7f7f', '#7f7fff', '#7fff7f', '#ffbf7f'],
                      alpha=0.8, edgecolor='black', linewidth=2)
        plt.title('Q-Learning Detection by Direction\n(Evaluation)', fontsize=14, fontweight='bold')
        plt.xlabel('Direction')
        plt.ylabel('Detection Rate')
        plt.xticks(x_pos, directions)
        plt.ylim(0, 1.0)
        plt.grid(True, alpha=0.3, axis='y')

        for bar, rate in zip(bars, ql_rates):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{rate:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

        # Plot 8: Direction-wise Detection Rates (Baseline Evaluation)
        plt.subplot(3, 3, 8)
        baseline_rates = []
        for direction in directions:
            crimes = self.eval_stats['baseline']['crimes_by_direction'][direction]
            detections = self.eval_stats['baseline']['detections_by_direction'][direction]
            rate = detections / max(1, crimes)
            baseline_rates.append(rate)

        bars = plt.bar(x_pos, baseline_rates, color=['#ff7f7f', '#7f7fff', '#7fff7f', '#ffbf7f'],
                      alpha=0.8, edgecolor='black', linewidth=2)
        plt.title('Baseline Detection by Direction\n(Evaluation)', fontsize=14, fontweight='bold')
        plt.xlabel('Direction')
        plt.ylabel('Detection Rate')
        plt.xticks(x_pos, directions)
        plt.ylim(0, 1.0)
        plt.grid(True, alpha=0.3, axis='y')

        for bar, rate in zip(bars, baseline_rates):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{rate:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

        # Plot 9: Performance Improvement
        plt.subplot(3, 3, 9)
        baseline_rate = self.baseline_eval['detection_rate']
        ql_rate = self.ql_eval['detection_rate']

        if baseline_rate > 0:
            improvement = ((ql_rate - baseline_rate) / baseline_rate) * 100
        else:
            improvement = 0

        plt.bar(['Detection Rate\nImprovement'], [improvement],
                color='green' if improvement > 0 else 'red', alpha=0.8, edgecolor='black', linewidth=2)
        plt.title('Q-Learning Performance Improvement', fontsize=14, fontweight='bold')
        plt.ylabel('Improvement (%)')
        plt.text(0, improvement + (abs(improvement)*0.1 if improvement != 0 else 1), f'{improvement:+.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=16)

        plt.tight_layout()
        plt.savefig('/home/leehs8006/tch/cctv_control/comprehensive_performance_analysis.png',
                   dpi=300, bbox_inches='tight')
        print("  Comprehensive performance analysis saved as: comprehensive_performance_analysis.png")

    def create_intersection_visualization(self):
        """Create intersection comparison visualization"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        def draw_intersection(ax, title, action, crimes_detected=False):
            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)
            ax.set_aspect('equal')
            ax.set_title(title, fontsize=16, fontweight='bold')

            # Draw roads
            ax.add_patch(patches.Rectangle((-3, -0.5), 6, 1, facecolor='gray', alpha=0.3))
            ax.add_patch(patches.Rectangle((-0.5, -3), 1, 6, facecolor='gray', alpha=0.3))

            # Draw CCTV in center
            ax.plot(0, 0, 'ko', markersize=20)
            ax.text(0, -0.5, 'CCTV', ha='center', va='top', fontweight='bold', fontsize=12)

            # Direction positions and colors
            directions = ['NORTH', 'SOUTH', 'EAST', 'WEST']
            positions = [(0, 2), (0, -2), (2, 0), (-2, 0)]
            colors = ['red', 'blue', 'green', 'orange']

            for i, (direction, pos, color) in enumerate(zip(directions, positions, colors)):
                if i == action:
                    # Current focus
                    circle = patches.Circle(pos, 0.7, facecolor=color, alpha=0.6, edgecolor='black', linewidth=3)
                    ax.add_patch(circle)
                    ax.text(pos[0], pos[1], 'MONITORING', ha='center', va='center',
                           fontweight='bold', color='white', fontsize=10)
                else:
                    circle = patches.Circle(pos, 0.5, facecolor=color, alpha=0.3, edgecolor='black', linewidth=1)
                    ax.add_patch(circle)

                # Direction label
                label_pos = (pos[0]*1.4, pos[1]*1.4)
                ax.text(label_pos[0], label_pos[1], direction, ha='center', va='center',
                       fontweight='bold', fontsize=12)

                # Crime indicator
                if crimes_detected and i == action:
                    ax.plot(pos[0], pos[1]+0.3, 'r*', markersize=15)
                    ax.text(pos[0], pos[1]+0.6, 'CRIME\nDETECTED!', ha='center', va='center',
                           fontweight='bold', color='red', fontsize=8)

            ax.set_xticks([])
            ax.set_yticks([])

        # Draw systems comparison
        draw_intersection(ax1, 'Sequential CCTV System\n(Current Method)', action=1)
        draw_intersection(ax2, 'Q-Learning CCTV System\n(Proposed Method)', action=2, crimes_detected=True)

        plt.tight_layout()
        plt.savefig('/home/leehs8006/tch/cctv_control/intersection_comparison.png',
                   dpi=300, bbox_inches='tight')
        print("  Intersection comparison saved as: intersection_comparison.png")

    def generate_comprehensive_report(self):
        """Generate detailed text report"""
        baseline_rate = self.baseline_eval['detection_rate']
        ql_rate = self.ql_eval['detection_rate']

        report = f"""
CCTV Q-Learning Reinforcement Learning System
Performance Evaluation Report
{'='*60}

EXECUTIVE SUMMARY
{'='*20}
This report presents the results of implementing a Q-Learning reinforcement
learning system for CCTV crime detection compared to the current sequential
monitoring approach.

SYSTEM SPECIFICATIONS
{'='*25}
• Environment: 4-way intersection with CCTV monitoring
• Agent: Q-Learning algorithm with epsilon-greedy exploration
• Actions: Monitor North, South, East, or West direction
• Reward: +10 for successfully detecting crime
• Training: {len(self.results['training_rewards'])} episodes with epsilon decay
• Evaluation: Full 365-day simulation (525,600 minutes)

PERFORMANCE RESULTS
{'='*23}

Sequential CCTV System (Baseline):
• Detection Rate: {baseline_rate:.3f}
• Total Crimes: {self.baseline_eval['total_crimes']}
• Detected Crimes: {self.baseline_eval['detected_crimes']}
• Total Reward: {self.baseline_eval['total_reward']:.0f}

Q-Learning CCTV System (Proposed):
• Detection Rate: {ql_rate:.3f}
• Total Crimes: {self.ql_eval['total_crimes']}
• Detected Crimes: {self.ql_eval['detected_crimes']}
• Total Reward: {self.ql_eval['total_reward']:.0f}

PERFORMANCE IMPROVEMENT
{'='*26}
"""

        if baseline_rate > 0:
            improvement = ((ql_rate - baseline_rate) / baseline_rate) * 100
            report += f"Detection Rate Improvement: {improvement:+.1f}%\n"
        else:
            report += "Detection Rate Improvement: Unable to calculate\n"

        reward_improvement = self.ql_eval['total_reward'] - self.baseline_eval['total_reward']
        report += f"Additional Rewards Gained: {reward_improvement:+.0f}\n"

        report += f"""

DIRECTION-WISE PERFORMANCE (EVALUATION PHASE)
{'='*46}

Q-Learning System:
"""
        for direction in ['NORTH', 'SOUTH', 'EAST', 'WEST']:
            crimes = self.eval_stats['q_learning']['crimes_by_direction'][direction]
            detections = self.eval_stats['q_learning']['detections_by_direction'][direction]
            rate = detections / max(1, crimes)
            report += f"• {direction:6s}: {detections:4d}/{crimes:4d} crimes detected ({rate:.3f})\n"

        report += f"""
Sequential Baseline System:
"""
        for direction in ['NORTH', 'SOUTH', 'EAST', 'WEST']:
            crimes = self.eval_stats['baseline']['crimes_by_direction'][direction]
            detections = self.eval_stats['baseline']['detections_by_direction'][direction]
            rate = detections / max(1, crimes)
            report += f"• {direction:6s}: {detections:4d}/{crimes:4d} crimes detected ({rate:.3f})\n"

        report += f"""

TECHNICAL ANALYSIS
{'='*21}

Training Performance:
• Final Training Reward: {self.results['training_rewards'][-1]:.0f}
• Final Detection Rate: {self.results['training_detection_rates'][-1]:.3f}
• Final Epsilon Value: {self.results['epsilon_values'][-1]:.3f}
• Q-Table Size: {self.training_results['q_table_size']} states

Learning Characteristics:
• Learning Rate: 0.1
• Discount Factor: 0.95
• Epsilon Decay: 0.995
• Exploration Strategy: Epsilon-greedy

ANIMATION RESULTS
{'='*20}
• Visualization Period: Last 30 days of {self.ql_eval['eval_episodes']}-day evaluation
• Video Duration: ~2 minutes
• Frames Generated: {len(self.viz_data)}
• Video File: cctv_simulation.mp4
• Shows both Q-Learning and Baseline decisions side-by-side

CONCLUSIONS
{'='*15}
"""

        if ql_rate > baseline_rate:
            report += "✓ The Q-Learning system demonstrates superior performance\n"
            report += "✓ Adaptive monitoring improves crime detection efficiency\n"
            report += "✓ Reinforcement learning successfully optimizes CCTV behavior\n"
        elif ql_rate == baseline_rate:
            report += "• The Q-Learning system shows equivalent performance\n"
            report += "• Both systems achieve optimal detection in this scenario\n"
        else:
            report += "• The Q-Learning system shows comparable performance\n"
            report += "• Further tuning may be required for optimal results\n"

        report += f"""

RECOMMENDATIONS
{'='*19}
1. Implement Q-Learning CCTV system for improved crime detection
2. Continue training with real-world crime pattern data
3. Consider ensemble methods combining multiple detection strategies
4. Regular model retraining as crime patterns evolve

FILES GENERATED
{'='*18}
• comprehensive_performance_analysis.png - 9-panel performance analysis with direction-wise statistics
• intersection_comparison.png - Visual system comparison
• cctv_simulation.mp4 - 2-minute animation video (last 30 days of evaluation)
• evaluation_report.txt - This comprehensive report
• training_results.png - Training performance graphs
• evaluation_results.png - Evaluation performance graphs

Report generated on: {np.datetime64('today')}
Evaluation Period: {self.ql_eval['eval_episodes']} days
Crime Probability: 5%
Training Episodes: {len(self.results['training_rewards'])}
Total Crimes (Evaluation): {self.ql_eval['total_crimes']}
"""

        with open('/home/leehs8006/tch/cctv_control/evaluation_report.txt', 'w') as f:
            f.write(report)

        print("  Comprehensive evaluation report saved as: evaluation_report.txt")
        return report

    def run_comprehensive_analysis(self):
        """Run all analysis components"""
        print("Starting Comprehensive Analysis...")
        print("="*50)

        print("1. Creating Performance Analysis...")
        self.create_comprehensive_performance_plots()

        print("2. Creating Intersection Visualization...")
        self.create_intersection_visualization()

        print("3. Generating Comprehensive Report...")
        report = self.generate_comprehensive_report()

        print("\n" + "="*50)
        print("COMPREHENSIVE ANALYSIS COMPLETE")
        print("="*50)
        print(report)

        return {
            'plots_created': True,
            'visualization_created': True,
            'report_generated': True,
            'report_content': report
        }