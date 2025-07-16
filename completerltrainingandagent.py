import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import pickle
import matplotlib.pyplot as plt
from supplychainenvi_rl_envi import SupplyChainEnv
import os

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.95, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Neural networks
        self.q_network = DQN(state_size, 128, action_size)
        self.target_network = DQN(state_size, 128, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.memory = ReplayBuffer(10000)
        self.batch_size = 64
        self.update_every = 4
        self.step_count = 0
        
        # Copy weights to target network
        self.update_target_network()
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def act(self, state, training=True):
        if training and random.random() <= self.epsilon:
            return random.choice(range(self.action_size))
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = self.memory.sample(self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
        
        self.step_count += 1
        if self.step_count % self.update_every == 0:
            self.update_target_network()
    
    def save(self, filepath):
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
    
    def load(self, filepath):
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']

def train_agent(episodes=1000, save_path="models/dqn_agent.pth"):
    """Train the DQN agent on the supply chain environment"""
    
    # Create environment
    env = SupplyChainEnv()
    state_size = len(env.reset())
    action_size = env.action_space.n
    
    # Create agent
    agent = DQNAgent(state_size, action_size)
    
    # Training metrics
    scores = []
    avg_scores = []
    episode_rewards = []
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    print(f"Training DQN Agent for {episodes} episodes...")
    print(f"State size: {state_size}, Action size: {action_size}")
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        step = 0
        
        while not done and step < 100:  # Max steps per episode
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            step += 1
            
            if len(agent.memory) > agent.batch_size:
                agent.replay()
        
        scores.append(total_reward)
        episode_rewards.append(total_reward)
        
        # Calculate moving average
        if len(scores) >= 100:
            avg_score = np.mean(scores[-100:])
            avg_scores.append(avg_score)
        else:
            avg_scores.append(np.mean(scores))
        
        # Print progress
        if episode % 100 == 0:
            print(f"Episode {episode}/{episodes}")
            print(f"Average Score (last 100): {avg_scores[-1]:.2f}")
            print(f"Epsilon: {agent.epsilon:.3f}")
            print(f"Total Steps: {env.total_steps}")
            print("-" * 50)
        
        # Save model periodically
        if episode % 500 == 0 and episode > 0:
            agent.save(save_path.replace('.pth', f'_episode_{episode}.pth'))
    
    # Save final model
    agent.save(save_path)
    
    # Save training metrics
    metrics = {
        'scores': scores,
        'avg_scores': avg_scores,
        'episode_rewards': episode_rewards
    }
    
    with open(save_path.replace('.pth', '_metrics.pkl'), 'wb') as f:
        pickle.dump(metrics, f)
    
    print(f"\nTraining completed!")
    print(f"Final average score: {avg_scores[-1]:.2f}")
    print(f"Model saved to: {save_path}")
    
    # Plot training progress
    plot_training_progress(scores, avg_scores, save_path.replace('.pth', '_training_plot.png'))
    
    return agent, metrics

def plot_training_progress(scores, avg_scores, save_path):
    """Plot and save training progress"""
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(scores, alpha=0.6, label='Episode Scores')
    plt.plot(avg_scores, label='Average Score (100 episodes)', color='red', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(avg_scores, color='red', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Average Score')
    plt.title('Average Score Trend')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Training plot saved to: {save_path}")

def evaluate_agent(model_path="models/dqn_agent.pth", episodes=10):
    """Evaluate a trained agent"""
    
    env = SupplyChainEnv()
    state_size = len(env.reset())
    action_size = env.action_space.n
    
    agent = DQNAgent(state_size, action_size)
    agent.load(model_path)
    
    total_rewards = []
    
    print(f"Evaluating agent for {episodes} episodes...")
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        step = 0
        
        while not done and step < 100:
            action = agent.act(state, training=False)  # No exploration
            state, reward, done, info = env.step(action)
            total_reward += reward
            step += 1
        
        total_rewards.append(total_reward)
        print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}")
    
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    
    print(f"\nEvaluation Results:")
    print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Best Episode: {max(total_rewards):.2f}")
    print(f"Worst Episode: {min(total_rewards):.2f}")
    
    return total_rewards

def compare_with_baseline():
    """Compare trained agent with random baseline"""
    
    print("Comparing trained agent with random baseline...")
    
    # Evaluate trained agent
    print("\n=== Trained Agent ===")
    trained_rewards = evaluate_agent()
    
    # Evaluate random agent
    print("\n=== Random Baseline ===")
    env = SupplyChainEnv()
    random_rewards = []
    
    for episode in range(10):
        state = env.reset()
        total_reward = 0
        done = False
        step = 0
        
        while not done and step < 100:
            action = env.action_space.sample()  # Random action
            state, reward, done, info = env.step(action)
            total_reward += reward
            step += 1
        
        random_rewards.append(total_reward)
        print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}")
    
    # Compare results
    print(f"\n=== Comparison ===")
    print(f"Trained Agent - Average: {np.mean(trained_rewards):.2f} ± {np.std(trained_rewards):.2f}")
    print(f"Random Agent - Average: {np.mean(random_rewards):.2f} ± {np.std(random_rewards):.2f}")
    
    improvement = (np.mean(trained_rewards) - np.mean(random_rewards)) / abs(np.mean(random_rewards)) * 100
    print(f"Improvement: {improvement:.1f}%")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train or evaluate DQN agent for supply chain management')
    parser.add_argument('--mode', choices=['train', 'evaluate', 'compare'], default='train',
                        help='Mode: train, evaluate, or compare')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of episodes for training')
    parser.add_argument('--model_path', default='models/dqn_agent.pth',
                        help='Path to save/load model')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        agent, metrics = train_agent(episodes=args.episodes, save_path=args.model_path)
    elif args.mode == 'evaluate':
        evaluate_agent(model_path=args.model_path)
    elif args.mode == 'compare':
        compare_with_baseline()
