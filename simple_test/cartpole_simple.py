import torch
import torch.nn as nn
import torch.optim as optim
import gym
import numpy as np
import matplotlib.pyplot as plt
from optimrl import GRPO, create_agent

# Simple policy network for CartPole
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.LogSoftmax(dim=-1)
        )
        
    def forward(self, x):
        return self.network(x)

def train(episodes=50):
    """Train a GRPO agent on CartPole-v1 for a few episodes"""
    env = gym.make('CartPole-v1')
    
    # Get environment dimensions
    state_dim = env.observation_space.shape[0]  # 4 for CartPole
    action_dim = env.action_space.n  # 2 for CartPole
    
    # Create policy network
    policy = PolicyNetwork(state_dim, action_dim)
    
    # Create GRPO agent
    agent = create_agent(
        "grpo",
        policy_network=policy,
        optimizer_class=optim.Adam,
        learning_rate=0.001,
        gamma=0.99,
        grpo_params={"epsilon": 0.2, "beta": 0.01},
        buffer_capacity=10000,
        batch_size=16
    )
    
    # Training loop
    rewards_history = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        # Single episode
        while not done:
            # Select action
            action = agent.act(state)
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store experience
            agent.store_experience(reward, done)
            
            # Update state and reward
            state = next_state
            episode_reward += reward
        
        # Update policy after episode
        result = agent.update()
        if result:
            loss, metrics = result
            print(f"Episode {episode+1}/{episodes}, Reward: {episode_reward}, Loss: {loss:.4f}")
        else:
            print(f"Episode {episode+1}/{episodes}, Reward: {episode_reward}, No update (insufficient data)")
        
        rewards_history.append(episode_reward)
    
    env.close()
    
    # Plot rewards
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_history)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('GRPO CartPole Training')
    plt.savefig('cartpole_rewards_simple.png')
    plt.show()
    
    print(f"Final average reward (last 10 episodes): {np.mean(rewards_history[-10:]):.2f}")
    
    return policy, rewards_history

if __name__ == "__main__":
    trained_policy, rewards = train(episodes=50)
    
    # Save the model
    torch.save(trained_policy.state_dict(), "cartpole_model_simple.pt")
    print("Model saved to cartpole_model_simple.pt") 