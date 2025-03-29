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
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.LogSoftmax(dim=-1)
        )
        
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        return self.network(x)

def train(episodes=100):
    """Train a GRPO agent on CartPole-v1 with improved hyperparameters"""
    env = gym.make('CartPole-v1')
    
    # Get environment dimensions
    state_dim = env.observation_space.shape[0]  # 4 for CartPole
    action_dim = env.action_space.n  # 2 for CartPole
    
    # Create policy network
    policy = PolicyNetwork(state_dim, action_dim)
    
    # Create GRPO agent with better hyperparameters
    agent = create_agent(
        "grpo",
        policy_network=policy,
        optimizer_class=optim.Adam,
        learning_rate=0.003,  # Higher learning rate
        gamma=0.99,
        grpo_params={"epsilon": 0.1, "beta": 0.01},  # Less clipping, same KL penalty
        buffer_capacity=10000,
        batch_size=8  # Smaller batch size for more frequent updates
    )
    
    # Training loop
    rewards_history = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        # Single episode
        steps = 0
        while not done:
            # Select action
            action = agent.act(state)
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Use a shaped reward to encourage longer episodes
            shaped_reward = reward * (1.0 + steps * 0.01)
            
            # Store experience
            agent.store_experience(shaped_reward, done)
            
            # Update state and reward
            state = next_state
            episode_reward += reward  # Track actual reward for reporting
            steps += 1
            
            # Update policy periodically during episode
            if steps % 10 == 0:
                agent.update()
        
        # Update policy after episode
        result = agent.update()
        if result:
            loss, metrics = result
            print(f"Episode {episode+1}/{episodes}, Reward: {episode_reward}, Loss: {loss:.6f}, Clip: {metrics['clip_fraction']:.4f}")
        else:
            print(f"Episode {episode+1}/{episodes}, Reward: {episode_reward}, No update (insufficient data)")
        
        rewards_history.append(episode_reward)
        
        # Early stopping if we solve CartPole
        if np.mean(rewards_history[-10:]) >= 475 and len(rewards_history) >= 10:
            print(f"Environment solved in {episode+1} episodes!")
            break
    
    env.close()
    
    # Plot rewards
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_history)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('GRPO CartPole Training (Improved)')
    plt.savefig('cartpole_rewards_improved.png')
    plt.show()
    
    print(f"Final average reward (last 10 episodes): {np.mean(rewards_history[-10:]):.2f}")
    
    return policy, rewards_history

if __name__ == "__main__":
    trained_policy, rewards = train(episodes=100)
    
    # Save the model
    torch.save(trained_policy.state_dict(), "cartpole_model_improved.pt")
    print("Model saved to cartpole_model_improved.pt") 