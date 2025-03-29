import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import matplotlib.pyplot as plt
from collections import deque
from optimrl.core import GRPO

# Implement our own replay buffer
class SimpleReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
        
    def add(self, state, action, reward, next_state, done, log_prob):
        self.buffer.append((state, action, reward, next_state, done, log_prob))
        
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
        
    def sample_recent(self, batch_size):
        # Sample from most recent experiences
        if batch_size >= len(self.buffer):
            return list(self.buffer)
        return list(self.buffer)[-batch_size:]
    
    def __len__(self):
        return len(self.buffer)

class EnhancedPolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(EnhancedPolicyNetwork, self).__init__()
        # Deeper network with more capacity
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        
    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        logits = self.network(x)
        return logits
    
    def get_action(self, state, deterministic=False):
        logits = self.forward(state)
        probs = torch.softmax(logits, dim=-1)
        
        if deterministic:
            action = torch.argmax(probs).item()
        else:
            action = torch.multinomial(probs, 1).item()
            
        return action, probs

def normalize_rewards(rewards, eps=1e-8):
    """Normalize rewards for better learning stability"""
    rewards = np.array(rewards)
    return (rewards - rewards.mean()) / (rewards.std() + eps)

def train():
    # Environment setup
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Hyperparameters
    gamma = 0.99
    learning_rate = 3e-4  # Reduced learning rate for stability
    epsilon = 0.2  # Increased from 0.1 to allow larger policy updates
    beta = 0.01  # KL divergence weight (beta in GRPO)
    buffer_size = 10000
    batch_size = 512  # Increased batch size for more stable updates
    max_episodes = 500  # Increased for more training time
    max_steps = 500
    update_interval = 10  # Update more frequently
    
    # Model setup
    policy = EnhancedPolicyNetwork(state_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    agent = GRPO(epsilon=epsilon, beta=beta)  # Using named parameters
    buffer = SimpleReplayBuffer(buffer_size)
    
    # Tracking metrics
    episode_rewards = []
    avg_rewards = []
    best_avg_reward = -float('inf')
    solved_reward = 475.0  # CartPole-v1 is considered solved when avg reward > 475 over 100 episodes
    
    # For early stopping
    recent_rewards = deque(maxlen=100)
    
    # Training loop
    for episode in range(1, max_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        states, actions, rewards, next_states, dones, log_probs = [], [], [], [], [], []
        
        for step in range(max_steps):
            # Select action
            action, action_probs = policy.get_action(state)
            action_log_prob = torch.log(action_probs[action])
            
            # Take action
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            
            # Store experience
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            log_probs.append(action_log_prob.detach())
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # Store episode in buffer
        for i in range(len(states)):
            buffer.add(states[i], actions[i], rewards[i], next_states[i], dones[i], log_probs[i])
        
        # Update policy if enough data
        if episode % update_interval == 0 and len(buffer) >= batch_size:
            # Sample batch from buffer
            batch = buffer.sample_recent(batch_size)
            
            # Unpack batch
            batch_states = np.array([e[0] for e in batch])
            batch_actions = np.array([e[1] for e in batch])
            batch_rewards = normalize_rewards([e[2] for e in batch])  # Normalize rewards
            batch_old_log_probs = torch.stack([e[5] for e in batch]).detach()
            
            # Using the current policy, get new log probabilities
            with torch.no_grad():
                logits = policy(torch.FloatTensor(batch_states))
                probs = torch.softmax(logits, dim=-1)
                batch_indices = torch.arange(len(batch_actions))
                new_log_probs = torch.log(probs[batch_indices, batch_actions])
            
            # Using the train_step method for optimization
            batch_data = {
                "log_probs_old": batch_old_log_probs,
                "log_probs_ref": batch_old_log_probs.clone(),  # Reference is old policy
                "log_probs_new": new_log_probs,
                "rewards": batch_rewards,
                "group_size": len(batch)
            }
            
            states_tensor = torch.FloatTensor(batch_states)
            loss, metrics = agent.train_step(policy, optimizer, batch_data, states_tensor)
            
            clip_fraction = metrics['clip_fraction']
            
            print(f"Episode {episode}/{max_episodes}, Reward: {episode_reward:.1f}, "
                  f"Avg: {np.mean(recent_rewards):.1f}, Loss: {loss:.6f}, Clip: {clip_fraction:.4f}")
        else:
            if episode == 1:
                print(f"Episode {episode}/{max_episodes}, Reward: {episode_reward:.1f}, No update (insufficient data)")
            else:
                print(f"Episode {episode}/{max_episodes}, Reward: {episode_reward:.1f}, "
                      f"Avg: {np.mean(recent_rewards):.1f}, No update")
        
        # Track progress
        episode_rewards.append(episode_reward)
        recent_rewards.append(episode_reward)
        
        if episode >= 100:
            avg_reward = np.mean(recent_rewards)
            avg_rewards.append(avg_reward)
            
            # Save best model
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                torch.save(policy.state_dict(), "cartpole_model_best.pt")
                
            # Check if solved
            if avg_reward >= solved_reward:
                print(f"\nEnvironment solved in {episode} episodes! Average reward: {avg_reward:.2f}")
                break
    
    # Print final performance
    if len(recent_rewards) >= 10:
        print(f"Final average reward (last 10 episodes): {np.mean(list(recent_rewards)[-10:]):.2f}")
    
    # Save final model
    torch.save(policy.state_dict(), "cartpole_model_tuned.pt")
    
    # Plot training progress
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards)
    if len(avg_rewards) > 0:
        plt.plot(range(99, 99 + len(avg_rewards)), avg_rewards, 'r')
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("CartPole-v1 Training Progress")
    plt.legend(["Episode Reward", "100-episode Average"])
    plt.savefig("cartpole_training_progress.png")
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