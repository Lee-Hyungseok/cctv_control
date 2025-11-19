import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random
import subprocess
from cctv_environment import CCTVEnvironment, Direction
from q_learning_agent import QLearningAgent
from baseline_cctv import BaselineCCTV
from typing import List, Dict, Tuple

class CCTVSimulation:
    def __init__(self, crime_probability: float = 0.05):
        self.crime_probability = crime_probability
        self.env = CCTVEnvironment(crime_probability)
        self.ql_agent = QLearningAgent(n_actions=4)
        self.baseline_agent = BaselineCCTV()

        # Simulation results
        self.ql_results = {'detection_rates': [], 'total_rewards': []}
        self.baseline_results = {'detection_rates': [], 'total_rewards': []}

        # Direction-wise statistics for training
        self.training_stats = {
            'q_learning': {
                'crimes_by_direction': {'NORTH': 0, 'SOUTH': 0, 'EAST': 0, 'WEST': 0},
                'detections_by_direction': {'NORTH': 0, 'SOUTH': 0, 'EAST': 0, 'WEST': 0}
            },
            'baseline': {
                'crimes_by_direction': {'NORTH': 0, 'SOUTH': 0, 'EAST': 0, 'WEST': 0},
                'detections_by_direction': {'NORTH': 0, 'SOUTH': 0, 'EAST': 0, 'WEST': 0}
            }
        }

        # Evaluation statistics
        self.eval_stats = {
            'q_learning': {
                'episode_rewards': [],
                'crimes_by_direction': {'NORTH': 0, 'SOUTH': 0, 'EAST': 0, 'WEST': 0},
                'detections_by_direction': {'NORTH': 0, 'SOUTH': 0, 'EAST': 0, 'WEST': 0}
            },
            'baseline': {
                'episode_rewards': [],
                'crimes_by_direction': {'NORTH': 0, 'SOUTH': 0, 'EAST': 0, 'WEST': 0},
                'detections_by_direction': {'NORTH': 0, 'SOUTH': 0, 'EAST': 0, 'WEST': 0}
            }
        }

        # Animation data
        self.animation_data = []

    def train_q_learning(self, episodes: int = 1000) -> Dict:
        print(f"Training Q-Learning agent for {episodes} episodes...")
        print("Also tracking Baseline performance during training...")

        for episode in range(episodes):
            # Q-Learning training episode
            state = self.env.reset()
            total_reward = 0

            while True:
                action = self.ql_agent.choose_action(state, training=True)

                # Track crimes before step
                direction_names = ['NORTH', 'SOUTH', 'EAST', 'WEST']
                for i, has_crime in enumerate(state['crimes']):
                    if has_crime:
                        self.training_stats['q_learning']['crimes_by_direction'][direction_names[i]] += 1
                        # Check if detected
                        if action == i:
                            self.training_stats['q_learning']['detections_by_direction'][direction_names[i]] += 1

                next_state, reward, done, _ = self.env.step(action)

                self.ql_agent.learn(state, action, reward, next_state, done)

                total_reward += reward
                state = next_state

                if done:
                    break

            # 에피소드 종료 시 epsilon 감소
            self.ql_agent.decay_epsilon()

            detection_rate = self.env.get_detection_probability()
            self.ql_results['detection_rates'].append(detection_rate)
            self.ql_results['total_rewards'].append(total_reward)

            # Baseline training episode (for comparison)
            state = self.env.reset()
            self.baseline_agent.reset()
            baseline_reward = 0

            while True:
                action = self.baseline_agent.choose_action(state)

                # Track crimes for baseline
                for i, has_crime in enumerate(state['crimes']):
                    if has_crime:
                        self.training_stats['baseline']['crimes_by_direction'][direction_names[i]] += 1
                        # Check if detected
                        if action == i:
                            self.training_stats['baseline']['detections_by_direction'][direction_names[i]] += 1

                next_state, reward, done, _ = self.env.step(action)

                baseline_reward += reward
                state = next_state

                if done:
                    break

            baseline_detection_rate = self.env.get_detection_probability()
            self.baseline_results['detection_rates'].append(baseline_detection_rate)
            self.baseline_results['total_rewards'].append(baseline_reward)

            if episode % 100 == 0:
                print(f"Episode {episode}:")
                print(f"  Q-Learning: Detection Rate = {detection_rate:.3f}, Total Reward = {total_reward}, Epsilon = {self.ql_agent.epsilon:.3f}")
                print(f"  Baseline:   Detection Rate = {baseline_detection_rate:.3f}, Total Reward = {baseline_reward}")

        return self.ql_agent.get_training_metrics()

    def evaluate_baseline(self, eval_episodes: int = 90) -> Dict:
        print(f"Evaluating baseline sequential CCTV for {eval_episodes} episodes (days)...")

        total_crimes_all = 0
        detected_crimes_all = 0
        total_rewards = []
        direction_names = ['NORTH', 'SOUTH', 'EAST', 'WEST']

        for episode in range(eval_episodes):
            state = self.env.reset()
            self.baseline_agent.reset()
            total_reward = 0

            while True:
                action = self.baseline_agent.choose_action(state)

                # Track crimes for baseline evaluation
                for i, has_crime in enumerate(state['crimes']):
                    if has_crime:
                        self.eval_stats['baseline']['crimes_by_direction'][direction_names[i]] += 1
                        # Check if detected
                        if action == i:
                            self.eval_stats['baseline']['detections_by_direction'][direction_names[i]] += 1

                next_state, reward, done, info = self.env.step(action)

                total_reward += reward
                state = next_state

                if done:
                    break

            total_crimes_all += info['total_crimes']
            detected_crimes_all += info['detected_crimes']
            total_rewards.append(total_reward)
            self.eval_stats['baseline']['episode_rewards'].append(total_reward)

            if episode % 10 == 0:
                print(f"Evaluation Day {episode+1}/{eval_episodes}: Crimes={info['total_crimes']}, Detected={info['detected_crimes']}")

        avg_detection_rate = detected_crimes_all / max(1, total_crimes_all)
        self.baseline_results['detection_rates'].append(avg_detection_rate)
        self.baseline_results['total_rewards'].append(sum(total_rewards))

        return {
            'detection_rate': avg_detection_rate,
            'total_reward': sum(total_rewards),
            'avg_reward_per_day': np.mean(total_rewards),
            'total_crimes': total_crimes_all,
            'detected_crimes': detected_crimes_all,
            'eval_episodes': eval_episodes
        }

    def evaluate_q_learning(self, eval_episodes: int = 90) -> Dict:
        print(f"Evaluating trained Q-Learning agent for {eval_episodes} episodes (days)...")

        total_crimes_all = 0
        detected_crimes_all = 0
        total_rewards = []
        direction_names = ['NORTH', 'SOUTH', 'EAST', 'WEST']

        for episode in range(eval_episodes):
            state = self.env.reset()
            total_reward = 0

            while True:
                action = self.ql_agent.choose_action(state, training=False)

                # Track crimes for Q-learning evaluation
                for i, has_crime in enumerate(state['crimes']):
                    if has_crime:
                        self.eval_stats['q_learning']['crimes_by_direction'][direction_names[i]] += 1
                        # Check if detected
                        if action == i:
                            self.eval_stats['q_learning']['detections_by_direction'][direction_names[i]] += 1

                next_state, reward, done, info = self.env.step(action)

                total_reward += reward
                state = next_state

                if done:
                    break

            total_crimes_all += info['total_crimes']
            detected_crimes_all += info['detected_crimes']
            total_rewards.append(total_reward)
            self.eval_stats['q_learning']['episode_rewards'].append(total_reward)

            if episode % 10 == 0:
                print(f"Evaluation Day {episode+1}/{eval_episodes}: Crimes={info['total_crimes']}, Detected={info['detected_crimes']}")

        avg_detection_rate = detected_crimes_all / max(1, total_crimes_all)

        return {
            'detection_rate': avg_detection_rate,
            'total_reward': sum(total_rewards),
            'avg_reward_per_day': np.mean(total_rewards),
            'total_crimes': total_crimes_all,
            'detected_crimes': detected_crimes_all,
            'eval_episodes': eval_episodes
        }

    def create_visualization_data(self, eval_episodes: int = 90, viz_days: int = 10) -> List[Dict]:
        # Create data for 2-minute visualization (only last 10 days)
        # 10 에피소드 * 144 스텝/에피소드 = 1440 스텝
        print(f"Creating visualization data for last {viz_days} days of {eval_episodes} episodes...")

        viz_data = []
        episode_num = 0
        start_episode = max(0, eval_episodes - viz_days)

        for episode in range(start_episode, eval_episodes):
            state = self.env.reset()
            self.baseline_agent.reset()

            step_in_episode = 0
            while True:
                # Q-Learning decision
                ql_action = self.ql_agent.choose_action(state, training=False)

                # Baseline decision
                baseline_action = self.baseline_agent.choose_action(state)

                # Current crimes from state
                crimes = {}
                for i, direction in enumerate(self.env.directions):
                    crimes[direction.name] = bool(state['crimes'][i])

                viz_data.append({
                    'episode': episode - start_episode,  # 0부터 시작하도록 조정
                    'step': step_in_episode,
                    'global_step': len(viz_data),
                    'ql_action': ql_action,
                    'baseline_action': baseline_action,
                    'crimes': crimes,
                    'ql_detection': crimes[list(crimes.keys())[ql_action]],
                    'baseline_detection': crimes[list(crimes.keys())[baseline_action]]
                })

                # Step environment (using Q-Learning action for state progression)
                state, _, done, _ = self.env.step(ql_action)
                step_in_episode += 1

                if done:
                    break

        print(f"Created {len(viz_data)} frames of visualization data")
        return viz_data

    def print_evaluation_statistics_table(self):
        """Print direction-wise crime and detection statistics for evaluation phase"""
        print("\n" + "="*80)
        print("EVALUATION PHASE - Direction-wise Crime and Detection Statistics")
        print("="*80)

        print("\n--- Q-Learning Agent ---")
        print(f"{'Direction':<10} {'Crimes':<15} {'Detections':<15} {'Detection Rate':<15}")
        print("-" * 60)

        ql_stats = self.eval_stats['q_learning']
        total_ql_crimes = sum(ql_stats['crimes_by_direction'].values())
        total_ql_detections = sum(ql_stats['detections_by_direction'].values())

        for direction in ['NORTH', 'SOUTH', 'EAST', 'WEST']:
            crimes = ql_stats['crimes_by_direction'][direction]
            detections = ql_stats['detections_by_direction'][direction]
            rate = detections / crimes if crimes > 0 else 0.0
            print(f"{direction:<10} {crimes:<15} {detections:<15} {rate:<15.3f}")

        ql_total_rate = total_ql_detections / total_ql_crimes if total_ql_crimes > 0 else 0.0
        print("-" * 60)
        print(f"{'TOTAL':<10} {total_ql_crimes:<15} {total_ql_detections:<15} {ql_total_rate:<15.3f}")

        print("\n--- Baseline (Sequential) Agent ---")
        print(f"{'Direction':<10} {'Crimes':<15} {'Detections':<15} {'Detection Rate':<15}")
        print("-" * 60)

        baseline_stats = self.eval_stats['baseline']
        total_baseline_crimes = sum(baseline_stats['crimes_by_direction'].values())
        total_baseline_detections = sum(baseline_stats['detections_by_direction'].values())

        for direction in ['NORTH', 'SOUTH', 'EAST', 'WEST']:
            crimes = baseline_stats['crimes_by_direction'][direction]
            detections = baseline_stats['detections_by_direction'][direction]
            rate = detections / crimes if crimes > 0 else 0.0
            print(f"{direction:<10} {crimes:<15} {detections:<15} {rate:<15.3f}")

        baseline_total_rate = total_baseline_detections / total_baseline_crimes if total_baseline_crimes > 0 else 0.0
        print("-" * 60)
        print(f"{'TOTAL':<10} {total_baseline_crimes:<15} {total_baseline_detections:<15} {baseline_total_rate:<15.3f}")

        print("\n--- Comparison (Q-Learning vs Baseline) ---")
        print(f"{'Direction':<10} {'QL Rate':<15} {'Baseline Rate':<15} {'Improvement':<15}")
        print("-" * 60)

        for direction in ['NORTH', 'SOUTH', 'EAST', 'WEST']:
            ql_crimes = ql_stats['crimes_by_direction'][direction]
            ql_detections = ql_stats['detections_by_direction'][direction]
            ql_rate = ql_detections / ql_crimes if ql_crimes > 0 else 0.0

            bl_crimes = baseline_stats['crimes_by_direction'][direction]
            bl_detections = baseline_stats['detections_by_direction'][direction]
            bl_rate = bl_detections / bl_crimes if bl_crimes > 0 else 0.0

            improvement = ((ql_rate - bl_rate) / bl_rate * 100) if bl_rate > 0 else 0.0
            print(f"{direction:<10} {ql_rate:<15.3f} {bl_rate:<15.3f} {improvement:+.2f}%")

        overall_improvement = ((ql_total_rate - baseline_total_rate) / baseline_total_rate * 100) if baseline_total_rate > 0 else 0.0
        print("-" * 60)
        print(f"{'OVERALL':<10} {ql_total_rate:<15.3f} {baseline_total_rate:<15.3f} {overall_improvement:+.2f}%")
        print("="*80 + "\n")

    def print_training_statistics_table(self):
        """Print direction-wise crime and detection statistics for training phase"""
        print("\n" + "="*80)
        print("TRAINING PHASE - Direction-wise Crime and Detection Statistics")
        print("="*80)

        print("\n--- Q-Learning Agent ---")
        print(f"{'Direction':<10} {'Crimes':<15} {'Detections':<15} {'Detection Rate':<15}")
        print("-" * 60)

        ql_stats = self.training_stats['q_learning']
        total_ql_crimes = sum(ql_stats['crimes_by_direction'].values())
        total_ql_detections = sum(ql_stats['detections_by_direction'].values())

        for direction in ['NORTH', 'SOUTH', 'EAST', 'WEST']:
            crimes = ql_stats['crimes_by_direction'][direction]
            detections = ql_stats['detections_by_direction'][direction]
            rate = detections / crimes if crimes > 0 else 0.0
            print(f"{direction:<10} {crimes:<15} {detections:<15} {rate:<15.3f}")

        ql_total_rate = total_ql_detections / total_ql_crimes if total_ql_crimes > 0 else 0.0
        print("-" * 60)
        print(f"{'TOTAL':<10} {total_ql_crimes:<15} {total_ql_detections:<15} {ql_total_rate:<15.3f}")

        print("\n--- Baseline (Sequential) Agent ---")
        print(f"{'Direction':<10} {'Crimes':<15} {'Detections':<15} {'Detection Rate':<15}")
        print("-" * 60)

        baseline_stats = self.training_stats['baseline']
        total_baseline_crimes = sum(baseline_stats['crimes_by_direction'].values())
        total_baseline_detections = sum(baseline_stats['detections_by_direction'].values())

        for direction in ['NORTH', 'SOUTH', 'EAST', 'WEST']:
            crimes = baseline_stats['crimes_by_direction'][direction]
            detections = baseline_stats['detections_by_direction'][direction]
            rate = detections / crimes if crimes > 0 else 0.0
            print(f"{direction:<10} {crimes:<15} {detections:<15} {rate:<15.3f}")

        baseline_total_rate = total_baseline_detections / total_baseline_crimes if total_baseline_crimes > 0 else 0.0
        print("-" * 60)
        print(f"{'TOTAL':<10} {total_baseline_crimes:<15} {total_baseline_detections:<15} {baseline_total_rate:<15.3f}")

        print("\n--- Comparison (Q-Learning vs Baseline) ---")
        print(f"{'Direction':<10} {'QL Rate':<15} {'Baseline Rate':<15} {'Improvement':<15}")
        print("-" * 60)

        for direction in ['NORTH', 'SOUTH', 'EAST', 'WEST']:
            ql_crimes = ql_stats['crimes_by_direction'][direction]
            ql_detections = ql_stats['detections_by_direction'][direction]
            ql_rate = ql_detections / ql_crimes if ql_crimes > 0 else 0.0

            bl_crimes = baseline_stats['crimes_by_direction'][direction]
            bl_detections = baseline_stats['detections_by_direction'][direction]
            bl_rate = bl_detections / bl_crimes if bl_crimes > 0 else 0.0

            improvement = ((ql_rate - bl_rate) / bl_rate * 100) if bl_rate > 0 else 0.0
            print(f"{direction:<10} {ql_rate:<15.3f} {bl_rate:<15.3f} {improvement:+.2f}%")

        overall_improvement = ((ql_total_rate - baseline_total_rate) / baseline_total_rate * 100) if baseline_total_rate > 0 else 0.0
        print("-" * 60)
        print(f"{'OVERALL':<10} {ql_total_rate:<15.3f} {baseline_total_rate:<15.3f} {overall_improvement:+.2f}%")
        print("="*80 + "\n")

    def plot_evaluation_results(self, eval_episodes: int):
        """Plot evaluation results comparing Q-Learning and Baseline"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Plot 1: Episode Rewards Comparison
        episodes_range = range(1, eval_episodes + 1)
        axes[0].plot(episodes_range, self.eval_stats['q_learning']['episode_rewards'],
                     label='Q-Learning', color='blue', alpha=0.7, linewidth=1.5)
        axes[0].plot(episodes_range, self.eval_stats['baseline']['episode_rewards'],
                     label='Baseline', color='red', alpha=0.7, linewidth=1.5)
        axes[0].set_title('Evaluation: Episode Rewards Comparison')
        axes[0].set_xlabel('Episode (Day)')
        axes[0].set_ylabel('Total Reward')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Smoothed Rewards (Moving Average)
        window = min(10, eval_episodes // 5)  # Adaptive window size
        if window > 1:
            ql_rewards_smooth = [np.mean(self.eval_stats['q_learning']['episode_rewards'][max(0, i-window):i+1])
                                 for i in range(len(self.eval_stats['q_learning']['episode_rewards']))]
            baseline_rewards_smooth = [np.mean(self.eval_stats['baseline']['episode_rewards'][max(0, i-window):i+1])
                                       for i in range(len(self.eval_stats['baseline']['episode_rewards']))]

            axes[1].plot(episodes_range, ql_rewards_smooth, label='Q-Learning (MA)', color='blue', linewidth=2)
            axes[1].plot(episodes_range, baseline_rewards_smooth, label='Baseline (MA)', color='red', linewidth=2)
            axes[1].set_title(f'Evaluation: Smoothed Rewards (Moving Avg, window={window})')
        else:
            axes[1].plot(episodes_range, self.eval_stats['q_learning']['episode_rewards'],
                         label='Q-Learning', color='blue', linewidth=2)
            axes[1].plot(episodes_range, self.eval_stats['baseline']['episode_rewards'],
                         label='Baseline', color='red', linewidth=2)
            axes[1].set_title('Evaluation: Episode Rewards')

        axes[1].set_xlabel('Episode (Day)')
        axes[1].set_ylabel('Average Reward')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('/home/leehs8006/tch/cctv_control/evaluation_results.png', dpi=300, bbox_inches='tight')
        print("Evaluation results plot saved as 'evaluation_results.png'")
        plt.show()

    def plot_training_results(self, training_metrics: Dict):
        fig, axes = plt.subplots(2, 3, figsize=(20, 10))

        # Plot 1: Episode Rewards Comparison (Q-Learning vs Baseline)
        axes[0, 0].plot(self.ql_results['total_rewards'], label='Q-Learning', color='blue', alpha=0.7)
        axes[0, 0].plot(self.baseline_results['total_rewards'], label='Baseline', color='red', alpha=0.7)
        axes[0, 0].set_title('Total Rewards per Episode - Training Comparison')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Plot 2: Detection Rates Comparison
        axes[0, 1].plot(self.ql_results['detection_rates'], label='Q-Learning', color='blue', alpha=0.7)
        axes[0, 1].plot(self.baseline_results['detection_rates'], label='Baseline', color='red', alpha=0.7)
        axes[0, 1].set_title('Detection Rate per Episode - Training Comparison')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Detection Rate')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Plot 3: Smoothed Rewards Comparison (Moving Average)
        window = 100
        ql_rewards_smooth = [np.mean(self.ql_results['total_rewards'][max(0, i-window):i+1])
                             for i in range(len(self.ql_results['total_rewards']))]
        baseline_rewards_smooth = [np.mean(self.baseline_results['total_rewards'][max(0, i-window):i+1])
                                   for i in range(len(self.baseline_results['total_rewards']))]

        axes[0, 2].plot(ql_rewards_smooth, label='Q-Learning (MA)', color='blue', linewidth=2)
        axes[0, 2].plot(baseline_rewards_smooth, label='Baseline (MA)', color='red', linewidth=2)
        axes[0, 2].set_title(f'Smoothed Rewards (Moving Avg, window={window})')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Average Reward')
        axes[0, 2].legend()
        axes[0, 2].grid(True)

        # Plot 4: Q-Learning Rewards Only
        axes[1, 0].plot(self.ql_results['total_rewards'], color='blue', alpha=0.6)
        axes[1, 0].set_title('Q-Learning Total Rewards')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Total Reward')
        axes[1, 0].grid(True)

        # Plot 5: Training Rewards (moving average)
        if len(training_metrics['rewards']) > 100:
            rewards = training_metrics['rewards']
            moving_avg = [np.mean(rewards[max(0, i-window):i+1]) for i in range(len(rewards))]
            axes[1, 1].plot(moving_avg, color='darkblue')
            axes[1, 1].set_title('Q-Learning Step-wise Moving Average Rewards')
            axes[1, 1].set_xlabel('Time Step')
            axes[1, 1].set_ylabel('Average Reward')
            axes[1, 1].grid(True)

        # Plot 6: Epsilon Decay
        if 'epsilon_history' in training_metrics and len(training_metrics['epsilon_history']) > 0:
            axes[1, 2].plot(training_metrics['epsilon_history'], color='green')
            axes[1, 2].set_title('Epsilon Value During Training')
            axes[1, 2].set_xlabel('Episode')
            axes[1, 2].set_ylabel('Epsilon')
            axes[1, 2].axhline(y=0.4, color='r', linestyle='--', label='Min Epsilon (0.4)')
            axes[1, 2].legend()
            axes[1, 2].grid(True)

        plt.tight_layout()
        plt.savefig('/home/leehs8006/tch/cctv_control/training_results.png', dpi=300, bbox_inches='tight')
        plt.show()

    def create_animation(self, viz_data: List[Dict]):
        print("Creating animation using ffmpeg streaming...")

        # Calculate FPS for 2-minute video (120 seconds)
        fps = max(int(len(viz_data) / 120), 1)
        print(f"Animation will use {fps} FPS for approximately 2 minutes playback")
        print(f"Total frames to render: {len(viz_data)}")

        # Setup ffmpeg process
        output_file = '/home/leehs8006/tch/cctv_control/cctv_simulation.mp4'

        ffmpeg_cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', '1600x800',  # 16x8 inches at 100 dpi
            '-pix_fmt', 'rgb24',
            '-r', str(fps),
            '-i', '-',  # Input from stdin
            '-an',  # No audio
            '-vcodec', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            output_file
        ]

        process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        # Direction colors
        direction_colors = {
            'NORTH': 'red',
            'SOUTH': 'blue',
            'EAST': 'green',
            'WEST': 'orange'
        }

        # Pre-calculate cumulative data for performance
        cumulative_ql_detections = []
        cumulative_baseline_detections = []
        cumulative_crimes = []

        ql_det_count = 0
        baseline_det_count = 0
        crime_count = 0

        for data in viz_data:
            ql_det_count += data['ql_detection']
            baseline_det_count += data['baseline_detection']
            crime_count += sum(data['crimes'].values())

            cumulative_ql_detections.append(ql_det_count)
            cumulative_baseline_detections.append(baseline_det_count)
            cumulative_crimes.append(crime_count)

        # Render frames one by one
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), dpi=100)

        try:
            for frame in range(len(viz_data)):
                if frame % 100 == 0:
                    print(f"Rendering frame {frame}/{len(viz_data)} ({100*frame/len(viz_data):.1f}%)")

                ax1.clear()
                ax2.clear()

                data = viz_data[frame]

                # Draw intersection
                ax1.set_xlim(-2, 2)
                ax1.set_ylim(-2, 2)
                ax1.set_aspect('equal')
                day_num = data['episode'] + 1
                ax1.set_title(f'CCTV Monitoring - Day {day_num}, Step {data["step"]+1}/144')

                # Draw roads
                ax1.plot([-2, 2], [0, 0], 'k-', linewidth=8, alpha=0.3)
                ax1.plot([0, 0], [-2, 2], 'k-', linewidth=8, alpha=0.3)

                # Draw CCTV in center
                ax1.plot(0, 0, 'ko', markersize=15)
                ax1.text(0, -0.3, 'CCTV', ha='center', va='top', fontweight='bold')

                # Draw direction areas
                directions = ['NORTH', 'SOUTH', 'EAST', 'WEST']
                positions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

                for i, (direction, pos) in enumerate(zip(directions, positions)):
                    color = direction_colors[direction]

                    # Crime indicator
                    if data['crimes'][direction]:
                        ax1.plot(pos[0], pos[1], 'ro', markersize=20, alpha=0.7)
                        ax1.text(pos[0], pos[1], 'CRIME', ha='center', va='center',
                                fontweight='bold', color='white', fontsize=8)

                    # Q-Learning CCTV focus
                    if data['ql_action'] == i:
                        circle1 = plt.Circle(pos, 0.3, color=color, alpha=0.5)
                        ax1.add_patch(circle1)
                        ax1.text(pos[0], pos[1]-0.6, 'Q-Learning\nFocus', ha='center', va='center',
                                fontweight='bold', color=color, fontsize=8)

                    # Direction label
                    ax1.text(pos[0]*1.5, pos[1]*1.5, direction, ha='center', va='center',
                            fontweight='bold', fontsize=10)

                # Performance comparison
                ax2.set_xlim(0, len(viz_data))
                ax2.set_ylim(0, 1)
                ax2.set_title('Cumulative Detection Performance')
                ax2.set_xlabel('Time Steps')
                ax2.set_ylabel('Detection Rate')

                # Calculate cumulative detection rates using pre-calculated data
                total_crimes = cumulative_crimes[frame]
                ql_detections = cumulative_ql_detections[frame]
                baseline_detections = cumulative_baseline_detections[frame]

                if total_crimes > 0:
                    ql_rate = ql_detections / total_crimes
                    baseline_rate = baseline_detections / total_crimes
                else:
                    ql_rate = 0
                    baseline_rate = 0

                # Calculate rates for plotting
                x_vals = list(range(frame+1))
                ql_rates = []
                baseline_rates = []

                for i in range(frame+1):
                    crimes_so_far = cumulative_crimes[i]
                    if crimes_so_far > 0:
                        ql_rates.append(cumulative_ql_detections[i] / crimes_so_far)
                        baseline_rates.append(cumulative_baseline_detections[i] / crimes_so_far)
                    else:
                        ql_rates.append(0)
                        baseline_rates.append(0)

                ax2.plot(x_vals, ql_rates, 'b-', label='Q-Learning', linewidth=2)
                ax2.plot(x_vals, baseline_rates, 'r-', label='Sequential', linewidth=2)
                ax2.legend()
                ax2.grid(True, alpha=0.3)

                # Add performance text
                ax2.text(0.02, 0.98, f'Q-Learning Rate: {ql_rate:.3f}\nSequential Rate: {baseline_rate:.3f}',
                        transform=ax2.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

                # Convert figure to RGB array
                fig.canvas.draw()
                # Use buffer_rgba() for modern matplotlib, then convert to RGB
                img_rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                img_rgba = img_rgba.reshape(fig.canvas.get_width_height()[::-1] + (4,))
                # Convert RGBA to RGB by removing alpha channel
                img_array = img_rgba[:, :, :3]

                # Write frame to ffmpeg
                try:
                    process.stdin.write(img_array.tobytes())
                except BrokenPipeError:
                    print("FFmpeg process terminated unexpectedly")
                    break

            print("Finalizing video encoding...")
            process.stdin.close()
            process.wait()

            if process.returncode == 0:
                print(f"Animation saved as cctv_simulation.mp4 ({len(viz_data)} frames at {fps} FPS)")
            else:
                stderr_output = process.stderr.read().decode()
                print(f"FFmpeg error: {stderr_output}")

        finally:
            plt.close(fig)
            if process.poll() is None:
                process.terminate()

        return None  # No animation object needed with direct ffmpeg encoding

def main(episodes=3000, eval_episodes=90, create_video=True):
    # Initialize simulation
    sim = CCTVSimulation(crime_probability=0.05)

    # Train Q-Learning agent
    print(f"Training Q-Learning agent for {episodes} episodes (days)...")
    training_metrics = sim.train_q_learning(episodes=episodes)
    print(f"\nTraining complete! Final epsilon: {training_metrics['epsilon']:.3f}")

    # Print training statistics table
    sim.print_training_statistics_table()

    # Plot training results
    print("\nGenerating training plots...")
    sim.plot_training_results(training_metrics)

    # Evaluate both systems
    print(f"\nEvaluating both systems for {eval_episodes} episodes (days)...")
    ql_eval = sim.evaluate_q_learning(eval_episodes=eval_episodes)
    baseline_eval = sim.evaluate_baseline(eval_episodes=eval_episodes)

    # Print evaluation statistics table
    sim.print_evaluation_statistics_table()

    # Plot evaluation results
    print("\nGenerating evaluation plots...")
    sim.plot_evaluation_results(eval_episodes)

    # Print comparison results
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON (90-DAY EVALUATION)")
    print("="*60)
    print(f"Q-Learning Detection Rate: {ql_eval['detection_rate']:.3f}")
    print(f"Sequential Detection Rate: {baseline_eval['detection_rate']:.3f}")

    if baseline_eval['detection_rate'] > 0:
        improvement = ((ql_eval['detection_rate'] - baseline_eval['detection_rate']) / baseline_eval['detection_rate'] * 100)
        print(f"Improvement: {improvement:+.1f}%")

    print(f"\nQ-Learning Total Reward: {ql_eval['total_reward']}")
    print(f"Sequential Total Reward: {baseline_eval['total_reward']}")
    print(f"\nQ-Learning Avg Reward/Day: {ql_eval['avg_reward_per_day']:.2f}")
    print(f"Sequential Avg Reward/Day: {baseline_eval['avg_reward_per_day']:.2f}")

    # Create visualization data for evaluation period (last 10 days only)
    print(f"\nCreating visualization for last 10 days of {eval_episodes} days evaluation...")
    viz_data = sim.create_visualization_data(eval_episodes=eval_episodes, viz_days=10)

    # Create video if requested
    animation_obj = None
    if create_video:
        print("\nCreating animation video (this may take several minutes)...")
        animation_obj = sim.create_animation(viz_data)

    # Return comprehensive results for report generation
    comprehensive_results = {
        'simulation': sim,
        'training_metrics': training_metrics,
        'q_learning_eval': ql_eval,
        'baseline_eval': baseline_eval,
        'visualization_data': viz_data,
        'animation': animation_obj,
        'training_rewards': sim.ql_results['total_rewards'],
        'training_detection_rates': sim.ql_results['detection_rates'],
        'epsilon_values': training_metrics.get('epsilon_history', [training_metrics['epsilon']] * len(sim.ql_results['total_rewards']))
    }

    return comprehensive_results

if __name__ == "__main__":
    results = main()