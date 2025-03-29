import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import matplotlib.pyplot as plt
from collections import deque

# Simple policy network
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        
    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        return self.network(x)
    
    def get_action(self, state, deterministic=False):
        logits = self.forward(state)
        probs = torch.softmax(logits, dim=-1)
        
        if deterministic:
            action = torch.argmax(probs).item()
        else:
            action = torch.multinomial(probs, 1).item()
            
        return action, probs

def discount_rewards(rewards, gamma=0.99):
    """Calculate discounted returns"""
    returns = []
    discounted_sum = 0
    
    for r in reversed(rewards):
        discounted_sum = r + gamma * discounted_sum
        returns.insert(0, discounted_sum)
    
    # Normalize returns
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
    return returns

def train():
    # Environment setup
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Hyperparameters
    learning_rate = 0.001
    gamma = 0.99
    max_episodes = 1000
    print_interval = 20
    
    # Model setup
    policy = PolicyNetwork(state_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    
    # Tracking metrics
    episode_rewards = []
    avg_rewards = []
    
    # For early stopping
    recent_rewards = deque(maxlen=100)
    solved_reward = 475.0
    best_avg_reward = -float('inf')
    
    # Training loop
    for episode in range(1, max_episodes + 1):
        state, _ = env.reset()
        rewards = []
        log_probs = []
        episode_reward = 0
        
        while True:
            # Select action
            action, probs = policy.get_action(state)
            log_prob = torch.log(probs[action])
            
            # Take action
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            
            # Record
            log_probs.append(log_prob)
            rewards.append(reward)
            episode_reward += reward
            
            state = next_state
            
            if done:
                break
        
        # Update policy
        returns = discount_rewards(rewards, gamma)
        policy_loss = []
        
        for log_prob, ret in zip(log_probs, returns):
            # Negative because we want to maximize rewards
            policy_loss.append(-log_prob * ret)
        
        # Backpropagation
        optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        optimizer.step()
        
        # Track progress
        episode_rewards.append(episode_reward)
        recent_rewards.append(episode_reward)
        
        if episode % print_interval == 0:
            avg_reward = np.mean(recent_rewards)
            print(f"Episode {episode}/{max_episodes}, Reward: {episode_reward}, Avg: {avg_reward:.1f}")
            
            if len(recent_rewards) >= 100:
                avg_rewards.append(avg_reward)
                
                # Save best model
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    torch.save(policy.state_dict(), "cartpole_model_best_pg.pt")
                
                # Check if solved
                if avg_reward >= solved_reward:
                    print(f"\nEnvironment solved in {episode} episodes! Average reward: {avg_reward:.2f}")
                    break
    
    # Print final performance
    if len(recent_rewards) >= 10:
        print(f"Final average reward (last 10 episodes): {np.mean(list(recent_rewards)[-10:]):.2f}")
    
    # Save final model
    torch.save(policy.state_dict(), "cartpole_model_pg.pt")
    
    # Plot training progress
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards)
    if len(avg_rewards) > 0:
        plt.plot(range(99, 99 + len(avg_rewards) * print_interval, print_interval), avg_rewards, 'r')
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("CartPole-v1 Training Progress")
    plt.legend(["Episode Reward", "100-episode Average"])
    plt.savefig("cartpole_training_progress_pg.png")
    plt.close()
    
    # Test the trained model
    print("\nTesting trained model:")
    test_episodes = 10
    test_rewards = []
    
    for i in range(test_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action, _ = policy.get_action(state, deterministic=True)
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            
            state = next_state
            episode_reward += reward
            
            # Render final test episode
            if i == test_episodes - 1:
                env.render()
                
        test_rewards.append(episode_reward)
    
    print(f"Average test reward: {np.mean(test_rewards):.2f}")
    env.close()
    
    return policy

if __name__ == "__main__":
    trained_policy = train() 