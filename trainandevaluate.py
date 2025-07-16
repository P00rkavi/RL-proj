#!/usr/bin/env python3
"""
Training and evaluation script for Supply Chain RL
Supports multiple RL algorithms and comprehensive evaluation
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# RL libraries
try:
    from stable_baselines3 import PPO, A2C, DQN, SAC
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.env_util import make_vec_env
except ImportError:
    print("Warning: stable-baselines3 not installed. Install with: pip install stable-baselines3")
    sys.exit(1)

from supplychainenvi_rl_envi import SupplyChainEnv

class SupplyChainTrainer:
    """Main training and evaluation class"""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.results_dir / "models").mkdir(exist_ok=True)
        (self.results_dir / "logs").mkdir(exist_ok=True)
        (self.results_dir / "plots").mkdir(exist_ok=True)
        
        self.algorithms = {
            'PPO': PPO,
            'A2C': A2C,
            'DQN': DQN,
            'SAC': SAC
        }
        
        self.trained_models = {}
        self.evaluation_results = {}
    
    def create_environment(self, monitor_file: Optional[str] = None):
        """Create and configure the environment"""
        env = SupplyChainEnv()
        
        if monitor_file:
            env = Monitor(env, monitor_file)
        
        return env
    
    def train_algorithm(self, 
                       algorithm: str, 
                       total_timesteps: int = 50000,
                       eval_freq: int = 5000,
                       save_freq: int = 10000) -> object:
        """Train a specific algorithm"""
        
        print(f"\n{'='*60}")
        print(f"Training {algorithm}")
        print(f"{'='*60}")
        
        # Create environment
        monitor_file = self.results_dir / "logs" / f"{algorithm}_monitor.csv"
        env = self.create_environment(str(monitor_file))
        
        # Create model
        model_class = self.algorithms[algorithm]
        
        if algorithm == 'DQN':
            # DQN requires discrete action space
            print("Note: DQN requires discrete action space")
            # You might need to modify your environment for DQN
            
        # Model parameters
        model_params = {
            'PPO': {
                'learning_rate': 3e-4,
                'n_steps': 2048,
                'batch_size': 64,
                'n_epochs': 10,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'verbose': 1
            },
            'A2C': {
                'learning_rate': 7e-4,
                'n_steps': 5,
                'gamma': 0.99,
                'gae_lambda': 1.0,
                'verbose': 1
            },
            'SAC': {
                'learning_rate': 3e-4,
                'buffer_size': 100000,
                'learning_starts': 1000,
                'batch_size': 256,
                'tau': 0.005,
                'gamma': 0.99,
                'verbose': 1
            }
        }
        
        if algorithm in model_params:
            model = model_class(env=env, **model_params[algorithm])
        else:
            model = model_class(env=env, verbose=1)
        
        # Setup callbacks
        eval_env = self.create_environment()
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(self.results_dir / "models" / f"{algorithm}_best"),
            log_path=str(self.results_dir / "logs" / f"{algorithm}_eval"),
            eval_freq=eval_freq,
            deterministic=True,
            render=False
        )
        
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=str(self.results_dir / "models" / f"{algorithm}_checkpoints"),
            name_prefix=f"{algorithm}_model"
        )
        
        # Train the model
        start_time = datetime.now()
        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, checkpoint_callback]
        )
        training_time = datetime.now() - start_time
        
        # Save final model
        model_path = self.results_dir / "models" / f"{algorithm}_final.zip"
        model.save(str(model_path))
        
        print(f"Training completed in {training_time}")
        print(f"Model saved to {model_path}")
        
        self.trained_models[algorithm] = model
        return model
    
    def evaluate_model(self, 
                      model, 
                      algorithm: str, 
                      n_eval_episodes: int = 10) -> Dict:
        """Evaluate a trained model"""
        
        print(f"\nEvaluating {algorithm}...")
        
        env = self.create_environment()
        
        # Evaluate policy
        mean_reward, std_reward = evaluate_policy(
            model, env, n_eval_episodes=n_eval_episodes, deterministic=True
        )
        
        # Collect detailed episode data
        episode_data = []
        for episode in range(n_eval_episodes):
            reset_result = env.reset()
            if isinstance(reset_result, tuple) and len(reset_result) == 2:
                obs, info = reset_result
            else:
                obs = reset_result
                info = {}
            episode_rewards = []
            episode_actions = []
            episode_observations = []
            
            total_reward = 0
            step_count = 0
            
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                step_result = env.step(action)
                if len(step_result) == 5:
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    obs, reward, done, info = step_result
                
                episode_rewards.append(reward)
                episode_actions.append(action)
                episode_observations.append(obs)
                
                total_reward += reward
                step_count += 1
                
            episode_data.append({
                'episode': episode,
                'total_reward': total_reward,
                'steps': step_count,
                'rewards': episode_rewards,
                'actions': episode_actions,
                'observations': episode_observations
            })
        
        results = {
            'algorithm': algorithm,
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'episode_data': episode_data,
            'n_episodes': n_eval_episodes
        }
        
        self.evaluation_results[algorithm] = results
        
        print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
        
        return results
    
    def train_all_algorithms(self, 
                           algorithms: List[str] = None,
                           total_timesteps: int = 50000) -> Dict:
        """Train multiple algorithms"""
        
        if algorithms is None:
            algorithms = ['PPO', 'A2C', 'SAC']  # Skip DQN for continuous action spaces
        
        results = {}
        
        for algorithm in algorithms:
            if algorithm not in self.algorithms:
                print(f"Warning: {algorithm} not supported")
                continue
                
            try:
                model = self.train_algorithm(algorithm, total_timesteps)
                eval_results = self.evaluate_model(model, algorithm)
                results[algorithm] = eval_results
                
            except Exception as e:
                print(f"Error training {algorithm}: {e}")
                continue
        
        return results
    
    def plot_training_results(self):
        """Plot training and evaluation results"""
        
        if not self.evaluation_results:
            print("No evaluation results to plot")
            return
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Algorithm Comparison', fontsize=16)
        
        algorithms = list(self.evaluation_results.keys())
        
        # Mean rewards comparison
        means = [self.evaluation_results[alg]['mean_reward'] for alg in algorithms]
        stds = [self.evaluation_results[alg]['std_reward'] for alg in algorithms]
        
        axes[0, 0].bar(algorithms, means, yerr=stds, capsize=5)
        axes[0, 0].set_title('Mean Episode Rewards')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Episode length comparison
        episode_lengths = []
        for alg in algorithms:
            lengths = [ep['steps'] for ep in self.evaluation_results[alg]['episode_data']]
            episode_lengths.append(lengths)
        
        axes[0, 1].boxplot(episode_lengths, labels=algorithms)
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Reward distribution
        for i, alg in enumerate(algorithms):
            rewards = [ep['total_reward'] for ep in self.evaluation_results[alg]['episode_data']]
            axes[1, 0].hist(rewards, alpha=0.6, label=alg, bins=10)
        
        axes[1, 0].set_title('Reward Distribution')
        axes[1, 0].set_xlabel('Total Reward')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Sample episode trajectory
        if algorithms:
            sample_alg = algorithms[0]
            sample_episode = self.evaluation_results[sample_alg]['episode_data'][0]
            
            axes[1, 1].plot(sample_episode['rewards'], label='Rewards')
            axes[1, 1].set_title(f'Sample Episode ({sample_alg})')
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Reward')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.results_dir / "plots" / "training_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plots saved to {plot_path}")
        
        plt.show()
    
    def save_results(self):
        """Save all results to files"""
        
        # Save evaluation results as JSON
        results_for_json = {}
        for alg, results in self.evaluation_results.items():
            # Convert numpy arrays to lists for JSON serialization
            json_results = {
                'algorithm': results['algorithm'],
                'mean_reward': float(results['mean_reward']),
                'std_reward': float(results['std_reward']),
                'n_episodes': results['n_episodes'],
                'episode_summaries': [
                    {
                        'episode': ep['episode'],
                        'total_reward': float(ep['total_reward']),
                        'steps': int(ep['steps'])
                    }
                    for ep in results['episode_data']
                ]
            }
            results_for_json[alg] = json_results
        
        json_path = self.results_dir / "evaluation_results.json"
        with open(json_path, 'w') as f:
            json.dump(results_for_json, f, indent=2)
        
        # Save detailed results as pickle
        pickle_path = self.results_dir / "detailed_results.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(self.evaluation_results, f)
        
        print(f"Results saved to {json_path} and {pickle_path}")
    
    def generate_report(self):
        """Generate a comprehensive report"""
        
        if not self.evaluation_results:
            print("No results to report")
            return
        
        report = []
        report.append("SUPPLY CHAIN RL TRAINING REPORT")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary table
        report.append("ALGORITHM PERFORMANCE SUMMARY")
        report.append("-" * 30)
        
        for alg, results in self.evaluation_results.items():
            report.append(f"{alg:>8}: {results['mean_reward']:>8.2f} ± {results['std_reward']:>6.2f}")
        
        report.append("")
        
        # Best algorithm
        best_alg = max(self.evaluation_results.keys(), 
                      key=lambda x: self.evaluation_results[x]['mean_reward'])
        report.append(f"BEST ALGORITHM: {best_alg}")
        report.append(f"Best Mean Reward: {self.evaluation_results[best_alg]['mean_reward']:.2f}")
        report.append("")
        
        # Detailed results
        report.append("DETAILED RESULTS")
        report.append("-" * 20)
        
        for alg, results in self.evaluation_results.items():
            report.append(f"\n{alg}:")
            report.append(f"  Episodes: {results['n_episodes']}")
            report.append(f"  Mean Reward: {results['mean_reward']:.2f}")
            report.append(f"  Std Reward: {results['std_reward']:.2f}")
            
            episode_rewards = [ep['total_reward'] for ep in results['episode_data']]
            report.append(f"  Min Reward: {min(episode_rewards):.2f}")
            report.append(f"  Max Reward: {max(episode_rewards):.2f}")
            
            episode_lengths = [ep['steps'] for ep in results['episode_data']]
            report.append(f"  Avg Episode Length: {np.mean(episode_lengths):.1f}")
        
        # Save report
        report_text = "\n".join(report)
        report_path = self.results_dir / "training_report.txt"
        
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        print(report_text)
        print(f"\nReport saved to {report_path}")

def main():
    """Main training function"""
    
    print("Supply Chain RL Training and Evaluation")
    print("=" * 50)
    
    # Configuration
    TOTAL_TIMESTEPS = 50000  # Adjust based on your needs
    ALGORITHMS = ['PPO', 'A2C', 'SAC']  # Algorithms to train
    
    # Create trainer
    trainer = SupplyChainTrainer()
    
    try:
        # Train all algorithms
        print("Starting training process...")
        results = trainer.train_all_algorithms(
            algorithms=ALGORITHMS,
            total_timesteps=TOTAL_TIMESTEPS
        )
        
        # Generate plots and reports
        trainer.plot_training_results()
        trainer.save_results()
        trainer.generate_report()
        
        print("\nTraining completed successfully!")
        print(f"Results saved in: {trainer.results_dir}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
