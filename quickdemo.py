#!/usr/bin/env python3
"""
Quick demonstration of the Supply Chain RL Environment
Run this to see the environment in action with basic policies
"""

import numpy as np
import matplotlib.pyplot as plt
from supplychainenvi_rl_envi import SupplyChainEnv
import time

def random_policy_demo():
    """Demonstrate random policy performance"""
    print("=" * 60)
    print("RANDOM POLICY DEMONSTRATION")
    print("=" * 60)
    
    env = SupplyChainEnv()
    obs, info = env.reset()
    
    episode_rewards = []
    inventory_levels = []
    orders = []
    demands = []
    
    total_reward = 0
    step_count = 0
    
    for step in range(100):
        # Random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        step_count += 1
        
        # Store metrics
        episode_rewards.append(reward)
        inventory_levels.append(obs[0])  # Assuming first obs is inventory
        orders.append(action)
        demands.append(info.get('demand', 0))
        
        if terminated or truncated:
            break
    
    print(f"Episode completed in {step_count} steps")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Average reward per step: {total_reward/step_count:.2f}")
    
    return {
        'rewards': episode_rewards,
        'inventory': inventory_levels,
        'orders': orders,
        'demands': demands,
        'total_reward': total_reward
    }

def simple_threshold_policy_demo():
    """Demonstrate simple threshold-based policy"""
    print("\n" + "=" * 60)
    print("THRESHOLD POLICY DEMONSTRATION")
    print("=" * 60)
    
    env = SupplyChainEnv()
    obs, info = env.reset()
    
    episode_rewards = []
    inventory_levels = []
    orders = []
    demands = []
    
    total_reward = 0
    step_count = 0
    
    # Simple threshold policy parameters
    reorder_point = 20
    order_quantity = 50
    
    for step in range(100):
        # Threshold-based policy
        current_inventory = obs[0] if len(obs) > 0 else 0
        
        if current_inventory <= reorder_point:
            action = order_quantity
        else:
            action = 0
        
        # Ensure action is within bounds
        action = np.clip(action, env.action_space.low, env.action_space.high)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        step_count += 1
        
        # Store metrics
        episode_rewards.append(reward)
        inventory_levels.append(obs[0] if len(obs) > 0 else 0)
        orders.append(action)
        demands.append(info.get('demand', 0))
        
        if terminated or truncated:
            break
    
    print(f"Episode completed in {step_count} steps")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Average reward per step: {total_reward/step_count:.2f}")
    
    return {
        'rewards': episode_rewards,
        'inventory': inventory_levels,
        'orders': orders,
        'demands': demands,
        'total_reward': total_reward
    }

def plot_comparison(random_results, threshold_results):
    """Plot comparison between policies"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Policy Comparison: Random vs Threshold', fontsize=16)
    
    steps_random = range(len(random_results['rewards']))
    steps_threshold = range(len(threshold_results['rewards']))
    
    # Rewards comparison
    axes[0, 0].plot(steps_random, random_results['rewards'], label='Random', alpha=0.7)
    axes[0, 0].plot(steps_threshold, threshold_results['rewards'], label='Threshold', alpha=0.7)
    axes[0, 0].set_title('Rewards per Step')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Inventory levels comparison
    axes[0, 1].plot(steps_random, random_results['inventory'], label='Random', alpha=0.7)
    axes[0, 1].plot(steps_threshold, threshold_results['inventory'], label='Threshold', alpha=0.7)
    axes[0, 1].set_title('Inventory Levels')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Inventory')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Orders comparison
    axes[1, 0].plot(steps_random, random_results['orders'], label='Random', alpha=0.7)
    axes[1, 0].plot(steps_threshold, threshold_results['orders'], label='Threshold', alpha=0.7)
    axes[1, 0].set_title('Order Quantities')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Order Size')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Cumulative rewards
    random_cum = np.cumsum(random_results['rewards'])
    threshold_cum = np.cumsum(threshold_results['rewards'])
    
    axes[1, 1].plot(steps_random, random_cum, label='Random', alpha=0.7)
    axes[1, 1].plot(steps_threshold, threshold_cum, label='Threshold', alpha=0.7)
    axes[1, 1].set_title('Cumulative Rewards')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Cumulative Reward')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def environment_info_demo():
    """Display environment information"""
    print("\n" + "=" * 60)
    print("ENVIRONMENT INFORMATION")
    print("=" * 60)
    
    env = SupplyChainEnv()
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    obs, info = env.reset()
    print(f"Initial observation shape: {np.array(obs).shape}")
    print(f"Initial observation: {obs}")
    print(f"Initial info: {info}")
    
    # Sample a few steps
    print("\nSample environment steps:")
    for i in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: Action={action}, Reward={reward:.2f}, Terminated={terminated}")

def main():
    """Main demonstration function"""
    print("Supply Chain RL Environment - Quick Demo")
    print("This demo shows basic environment functionality")
    
    try:
        # Show environment info
        environment_info_demo()
        
        # Run random policy
        print("\nRunning random policy demo...")
        random_results = random_policy_demo()
        
        # Run threshold policy
        print("\nRunning threshold policy demo...")
        threshold_results = simple_threshold_policy_demo()
        
        # Compare results
        print("\n" + "=" * 60)
        print("POLICY COMPARISON")
        print("=" * 60)
        print(f"Random Policy Total Reward: {random_results['total_reward']:.2f}")
        print(f"Threshold Policy Total Reward: {threshold_results['total_reward']:.2f}")
        
        improvement = threshold_results['total_reward'] - random_results['total_reward']
        print(f"Improvement: {improvement:.2f}")
        
        # Plot results
        print("\nGenerating comparison plots...")
        plot_comparison(random_results, threshold_results)
        
        print("\nDemo completed successfully!")
        print("Try running train_and_evaluate.py for advanced RL training")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        print("Make sure supplychainenv.py is in the same directory")
        print("and all required packages are installed (run: pip install -e .)")

if __name__ == "__main__":
    main()
