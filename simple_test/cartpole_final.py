import torch
import torch.nn as nn
import torch.optim as optim
import gym
import numpy as np
import matplotlib.pyplot as plt
from optimrl import GRPO, create_agent

# Policy network for CartPole
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, output_dim),
            nn.LogSoftmax(dim=-1)
        )
        
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        return self.network(x)

def train(episodes=200):
    """Train a GRPO agent on CartPole-v1 with fixed algorithm"""
    env = gym.make('CartPole-v1')
    
    # Get environment dimensions
    state_dim = env.observation_space.shape[0]  # 4 for CartPole
    action_dim = env.action_space.n  # 2 for CartPole
    
    # Create policy network
    policy = PolicyNetwork(state_dim, action_dim)
    
    # Create GRPO agent with optimized hyperparameters
    agent = create_agent(
        "grpo",
        policy_network=policy,
        optimizer_class=optim.Adam,
        learning_rate=0.005,  # Higher learning rate
        gamma=0.99,
        grpo_params={"epsilon": 0.2, "beta": 0.005},  # Standard clipping, lower KL penalty
        buffer_capacity=10000,
        batch_size=32  # Larger batch size for more stable updates
    )
    
    # Training loop
    rewards_history = []
    avg_rewards = []
    
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
            
            # Store experience with actual reward
            agent.store_experience(reward, done)
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            steps += 1
        
        # Update policy after episode
        result = agent.update()
        if result:
            loss, metrics = result
            clip_fraction = metrics['clip_fraction']
            avg_reward = np.mean(rewards_history[-10:]) if len(rewards_history) >= 10 else episode_reward
            print(f"Episode {episode+1}/{episodes}, Reward: {episode_reward}, Avg: {avg_reward:.1f}, Loss: {loss:.6f}, Clip: {clip_fraction:.4f}")
        else:
            print(f"Episode {episode+1}/{episodes}, Reward: {episode_reward}, No update (insufficient data)")
        
        rewards_history.append(episode_reward)
        
        # Track average reward for plotting
        if len(rewards_history) >= 10:
            avg_rewards.append(np.mean(rewards_history[-10:]))
        else:
            avg_rewards.append(episode_reward)
        
        # Early stopping if we solve CartPole (average reward of 475+ over 10 episodes)
        if np.mean(rewards_history[-10:]) >= 475 and len(rewards_history) >= 10:
            print(f"Environment solved in {episode+1} episodes!")
            break
    
    env.close()
    
    # Plot rewards
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(rewards_history)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Episode Rewards')
    
    plt.subplot(1, 2, 2)
    plt.plot(avg_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward (10 episodes)')
    plt.title('Learning Progress')
    
    plt.tight_layout()
    plt.savefig('cartpole_rewards_final.png')
    plt.show()
    
    print(f"Final average reward (last 10 episodes): {np.mean(rewards_history[-10:]):.2f}")
    
    return policy, rewards_history

if __name__ == "__main__":
    trained_policy, rewards = train(episodes=200)
    
    # Save the model
    torch.save(trained_policy.state_dict(), "cartpole_model_final.pt")
    print("Model saved to cartpole_model_final.pt")
    
    # Test the trained model
    print("\nTesting trained model:")
    env = gym.make('CartPole-v1', render_mode='human')
    state, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # Get action from trained policy
        state_tensor = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            log_probs = trained_policy(state_tensor)
            action = torch.argmax(log_probs).item()
        
        # Take action
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
    
    print(f"Test reward: {total_reward}")
    env.close() 