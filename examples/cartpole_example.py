#!/usr/bin/env python
# Example of training a GRPO agent on the CartPole environment

import torch
import torch.nn as nn
import torch.optim as optim
import gym
import numpy as np
import matplotlib.pyplot as plt
from optimrl import GRPO, GRPOAgent, create_agent

# Define a simple policy network for CartPole
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.LogSoftmax(dim=-1)
        )
        
    def forward(self, x):
        return self.network(x)

def train_cartpole(episodes=500, render=False):
    # Create the CartPole environment
    env = gym.make('CartPole-v1')
    
    # Get environment dimensions
    state_dim = env.observation_space.shape[0]  # 4 for CartPole
    action_dim = env.action_space.n  # 2 for CartPole
    
    # Create the policy network
    policy_network = PolicyNetwork(state_dim, action_dim)
    
    # Initialize the GRPO agent
    agent = create_agent(
        "grpo",
        policy_network=policy_network,
        optimizer_class=optim.Adam,
        learning_rate=0.001,
        gamma=0.99,
        grpo_params={"epsilon": 0.2, "beta": 0.01},
        buffer_capacity=10000,
        batch_size=32
    )
    
    # Training loop
    rewards_history = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            if render and episode % 50 == 0:
                env.render()
                
            # Select an action
            action = agent.act(state)
            
            # Take the action in the environment
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            
            # Store experience and update policy
            agent.store_experience(reward, done)
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            
        # Update policy after episode ends
        agent.update()
        
        # Record rewards
        rewards_history.append(episode_reward)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            print(f"Episode {episode + 1}/{episodes} | Avg Reward: {avg_reward:.2f}")
            
    env.close()
    
    # Plot rewards
    plt.figure(figsize=(10, 6))
    plt.plot(rewards_history)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('GRPO on CartPole-v1')
    plt.savefig('cartpole_rewards.png')
    plt.show()
    
    return rewards_history, policy_network

if __name__ == "__main__":
    rewards, model = train_cartpole(episodes=300, render=False)
    print("Training completed!")
    
    # Save the trained model
    torch.save(model.state_dict(), "cartpole_grpo_model.pt")
    print("Model saved to cartpole_grpo_model.pt") 