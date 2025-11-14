import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import random
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

        # Animation data
        self.animation_data = []

    def train_q_learning(self, episodes: int = 1000) -> Dict:
        print(f"Training Q-Learning agent for {episodes} episodes...")

        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0

            while True:
                action = self.ql_agent.choose_action(state, training=True)
                next_state, reward, done, info = self.env.step(action)

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

            if episode % 100 == 0:
                print(f"Episode {episode}: Detection Rate = {detection_rate:.3f}, Total Reward = {total_reward}, Epsilon = {self.ql_agent.epsilon:.3f}")

        return self.ql_agent.get_training_metrics()

    def evaluate_baseline(self, eval_episodes: int = 90) -> Dict:
        print(f"Evaluating baseline sequential CCTV for {eval_episodes} episodes (days)...")

        total_crimes_all = 0
        detected_crimes_all = 0
        total_rewards = []

        for episode in range(eval_episodes):
            state = self.env.reset()
            self.baseline_agent.reset()
            total_reward = 0

            while True:
                action = self.baseline_agent.choose_action(state)
                next_state, reward, done, info = self.env.step(action)

                total_reward += reward
                state = next_state

                if done:
                    break

            total_crimes_all += info['total_crimes']
            detected_crimes_all += info['detected_crimes']
            total_rewards.append(total_reward)

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

        for episode in range(eval_episodes):
            state = self.env.reset()
            total_reward = 0

            while True:
                action = self.ql_agent.choose_action(state, training=False)
                next_state, reward, done, info = self.env.step(action)

                total_reward += reward
                state = next_state

                if done:
                    break

            total_crimes_all += info['total_crimes']
            detected_crimes_all += info['detected_crimes']
            total_rewards.append(total_reward)

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

    def create_visualization_data(self, eval_episodes: int = 90) -> List[Dict]:
        # Create data for 10-minute visualization (90 episodes = 90 days)
        # 90 에피소드 * 144 스텝/에피소드 = 12960 스텝
        print(f"Creating visualization data for {eval_episodes} episodes...")

        viz_data = []
        episode_num = 0

        for episode in range(eval_episodes):
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
                    'episode': episode,
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

    def plot_training_results(self, training_metrics: Dict):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: Episode Rewards
        axes[0, 0].plot(self.ql_results['total_rewards'])
        axes[0, 0].set_title('Total Rewards per Episode')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].grid(True)

        # Plot 2: Detection Rates
        axes[0, 1].plot(self.ql_results['detection_rates'])
        axes[0, 1].set_title('Detection Rate per Episode')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Detection Rate')
        axes[0, 1].grid(True)

        # Plot 3: Training Rewards (moving average)
        if len(training_metrics['rewards']) > 100:
            rewards = training_metrics['rewards']
            window = 100
            moving_avg = [np.mean(rewards[max(0, i-window):i+1]) for i in range(len(rewards))]
            axes[1, 0].plot(moving_avg)
            axes[1, 0].set_title('Moving Average Rewards (window=100)')
            axes[1, 0].set_xlabel('Time Step')
            axes[1, 0].set_ylabel('Average Reward')
            axes[1, 0].grid(True)

        # Plot 4: Epsilon Decay
        if 'epsilon_history' in training_metrics and len(training_metrics['epsilon_history']) > 0:
            axes[1, 1].plot(training_metrics['epsilon_history'])
            axes[1, 1].set_title('Epsilon Value During Training')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Epsilon')
            axes[1, 1].axhline(y=0.4, color='r', linestyle='--', label='Min Epsilon (0.4)')
            axes[1, 1].legend()
            axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig('/home/leehs8006/tch/cctv_control/training_results.png', dpi=300, bbox_inches='tight')
        plt.show()

    def create_animation(self, viz_data: List[Dict]):
        print("Creating animation...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # CCTV intersection visualization
        intersection_size = 4

        # Direction colors
        direction_colors = {
            'NORTH': 'red',
            'SOUTH': 'blue',
            'EAST': 'green',
            'WEST': 'orange'
        }

        def animate(frame):
            ax1.clear()
            ax2.clear()

            if frame >= len(viz_data):
                return

            data = viz_data[frame]

            # Draw intersection
            ax1.set_xlim(-2, 2)
            ax1.set_ylim(-2, 2)
            ax1.set_aspect('equal')
            day_num = data['episode'] + 1
            ax1.set_title(f'CCTV Monitoring - Day {day_num}, Step {data["step"]+1}/144')

            # Draw roads
            ax1.plot([-2, 2], [0, 0], 'k-', linewidth=8, alpha=0.3)  # East-West road
            ax1.plot([0, 0], [-2, 2], 'k-', linewidth=8, alpha=0.3)  # North-South road

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

            # Calculate cumulative detection rates
            ql_detections = sum([d['ql_detection'] for d in viz_data[:frame+1]])
            baseline_detections = sum([d['baseline_detection'] for d in viz_data[:frame+1]])
            total_crimes = sum([sum(d['crimes'].values()) for d in viz_data[:frame+1]])

            if total_crimes > 0:
                ql_rate = ql_detections / total_crimes
                baseline_rate = baseline_detections / total_crimes
            else:
                ql_rate = 0
                baseline_rate = 0

            x_vals = list(range(frame+1))
            ql_rates = []
            baseline_rates = []

            for i in range(frame+1):
                crimes_so_far = sum([sum(d['crimes'].values()) for d in viz_data[:i+1]])
                ql_det_so_far = sum([d['ql_detection'] for d in viz_data[:i+1]])
                base_det_so_far = sum([d['baseline_detection'] for d in viz_data[:i+1]])

                if crimes_so_far > 0:
                    ql_rates.append(ql_det_so_far / crimes_so_far)
                    baseline_rates.append(base_det_so_far / crimes_so_far)
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

        # Create animation
        # 90 에피소드 * 144 스텝 = 12960 프레임
        # 10분(600초) 안에 재생하려면 fps = 12960/600 ≈ 22
        fps = max(int(len(viz_data) / 600), 1)  # 최소 1 fps
        print(f"Animation will use {fps} FPS for approximately 10 minutes playback")

        ani = animation.FuncAnimation(fig, animate, frames=len(viz_data),
                                    interval=1000/fps, repeat=True)

        # Save animation as MP4
        print("Saving animation as MP4 (this may take a few minutes)...")
        ani.save('/home/leehs8006/tch/cctv_control/cctv_simulation.mp4',
                writer='ffmpeg', fps=fps, bitrate=1800)
        print(f"Animation saved as cctv_simulation.mp4 ({len(viz_data)} frames at {fps} FPS)")

        plt.show()
        return ani

def main(episodes=1000, eval_episodes=90, create_video=True):
    # Initialize simulation
    sim = CCTVSimulation(crime_probability=0.05)

    # Train Q-Learning agent
    print(f"Training Q-Learning agent for {episodes} episodes (days)...")
    training_metrics = sim.train_q_learning(episodes=episodes)
    print(f"\nTraining complete! Final epsilon: {training_metrics['epsilon']:.3f}")

    # Plot training results
    print("\nGenerating training plots...")
    sim.plot_training_results(training_metrics)

    # Evaluate both systems
    print(f"\nEvaluating both systems for {eval_episodes} episodes (days)...")
    ql_eval = sim.evaluate_q_learning(eval_episodes=eval_episodes)
    baseline_eval = sim.evaluate_baseline(eval_episodes=eval_episodes)

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

    # Create visualization data for evaluation period
    print(f"\nCreating visualization for {eval_episodes} days...")
    viz_data = sim.create_visualization_data(eval_episodes=eval_episodes)

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