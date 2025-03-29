#!/usr/bin/env python
# Example of training a ContinuousGRPOAgent on the Pendulum environment

import torch
import torch.nn as nn
import torch.optim as optim
import gym
import numpy as np
import matplotlib.pyplot as plt
from optimrl import GRPO, ContinuousGRPOAgent, create_agent

# Define a continuous policy network for Pendulum
class ContinuousPolicyNetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # Output both mean and log_std for each action dimension
        self.output_layer = nn.Linear(64, action_dim * 2)
        
    def forward(self, x):
        x = self.shared_layers(x)
        output = self.output_layer(x)
        return output

def train_pendulum(episodes=300, render=False):
    # Create the Pendulum environment
    env = gym.make('Pendulum-v1')
    
    # Get environment dimensions
    state_dim = env.observation_space.shape[0]  # 3 for Pendulum
    action_dim = env.action_space.shape[0]  # 1 for Pendulum
    
    # Action bounds
    action_bounds = (env.action_space.low[0], env.action_space.high[0])
    
    # Create the policy network
    policy_network = ContinuousPolicyNetwork(state_dim, action_dim)
    
    # Initialize the Continuous GRPO agent
    agent = create_agent(
        "continuous_grpo",
        policy_network=policy_network,
        optimizer_class=optim.Adam,
        action_dim=action_dim,
        learning_rate=0.0005,
        gamma=0.99,
        grpo_params={"epsilon": 0.2, "beta": 0.01},
        buffer_capacity=10000,
        batch_size=64,
        min_std=0.01,
        action_bounds=action_bounds
    )
    
    # Training loop
    rewards_history = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        while not done and step < 200:  # Max steps for Pendulum is typically 200
            if render and episode % 50 == 0:
                env.render()
                
            # Select an action
            action = agent.act(state)
            
            # Take the action in the environment
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            
            # Store experience
            agent.store_experience(reward, done)
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            step += 1
            
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
    plt.title('GRPO on Pendulum-v1')
    plt.savefig('pendulum_rewards.png')
    plt.show()
    
    return rewards_history, policy_network

if __name__ == "__main__":
    rewards, model = train_pendulum(episodes=300, render=False)
    print("Training completed!")
    
    # Save the trained model
    torch.save(model.state_dict(), "pendulum_grpo_model.pt")
    print("Model saved to pendulum_grpo_model.pt") 