# optimrl/agents.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any, Dict, List, Tuple, Optional, Union
from collections import deque
import random
from .core import GRPO

class ExperienceBuffer:
    """
    A buffer to store and sample experiences for experience replay.
    
    Experience replay allows the agent to learn from past experiences multiple times,
    improving sample efficiency and reducing correlation between consecutive samples.
    """
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize the experience buffer.
        
        Args:
            capacity (int): Maximum number of experiences to store in the buffer.
                When buffer is full, oldest experiences are replaced.
        """
        self.buffer = deque(maxlen=capacity)
        
    def add(self, experience: Dict[str, Any]) -> None:
        """
        Add an experience to the buffer.
        
        Args:
            experience (dict): A dictionary containing experience data with keys:
                - 'state': The environment state
                - 'action': The action taken
                - 'reward': The reward received
                - 'log_prob': Log probability of the action
                - Any other relevant information
        """
        self.buffer.append(experience)
        
    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        """
        Sample a batch of experiences from the buffer.
        
        Args:
            batch_size (int): Number of experiences to sample
            
        Returns:
            list: A list of experience dictionaries
            
        Note:
            If the buffer contains fewer than batch_size experiences,
            all experiences will be returned.
        """
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.buffer)
        
    def clear(self) -> None:
        """Clear all experiences from the buffer."""
        self.buffer.clear()

class BaseAgent:
    """Abstract base agent class."""
    def act(self, state: Any) -> int:
        raise NotImplementedError

    def update(self, reward: float) -> Any:
        raise NotImplementedError

class GRPOAgent(BaseAgent):
    def __init__(self, 
                 policy_network: nn.Module, 
                 optimizer_class: Any,
                 learning_rate: float = 0.01, 
                 gamma: float = 0.99, 
                 grpo_params: Dict[str, float] = None,
                 buffer_capacity: int = 10000,
                 batch_size: int = 64):
        """
        Initialize a Group Relative Policy Optimization agent.
        
        Args:
            policy_network (nn.Module): Neural network that outputs action log probabilities
            optimizer_class: PyTorch optimizer class (e.g., torch.optim.Adam)
            learning_rate (float): Learning rate for the optimizer
            gamma (float): Discount factor for future rewards
            grpo_params (dict): Parameters for the GRPO algorithm (epsilon, beta)
            buffer_capacity (int): Maximum size of the experience replay buffer
            batch_size (int): Number of experiences to sample for each update
        """
        self.policy_network = policy_network
        self.optimizer = optimizer_class(self.policy_network.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.grpo = GRPO(**(grpo_params if grpo_params is not None else {}))
        self.memory = []  # To store recent experiences
        self.experience_buffer = ExperienceBuffer(capacity=buffer_capacity)
        self.batch_size = batch_size

    def act(self, state: Any) -> int:
        """
        Select an action based on the current state.
        
        Args:
            state: The current environment state
            
        Returns:
            int: The selected action index
        """
        state_tensor = torch.tensor(state, dtype=torch.float32)
        # Forward pass through the policy network
        log_probs = self.policy_network(state_tensor)
        probs = torch.exp(log_probs)
        # Sample an action from the probabilities
        action = torch.multinomial(probs, num_samples=1).item()
        # Store the log probability for updating later
        self.memory.append({
            'state': state_tensor,
            'action': action,
            'log_prob': log_probs[action]
        })
        return action

    def store_experience(self, reward: float, done: bool = False) -> None:
        """
        Store the most recent experience with its reward.
        
        Args:
            reward (float): The reward received for the last action
            done (bool): Whether the episode has ended
        """
        if not self.memory:
            return
            
        experience = self.memory[-1]
        experience['reward'] = reward
        experience['done'] = done
        
        # Add to experience buffer for replay
        self.experience_buffer.add(experience.copy())
        
    def update(self, reward: float = None) -> Any:
        """
        Update the policy network using experience replay.
        
        Args:
            reward (float, optional): If provided, stores this reward before updating
            
        Returns:
            tuple or None: (loss, metrics) if update was performed, None otherwise
        """
        # If reward is provided, store the experience first
        if reward is not None:
            self.store_experience(reward)
            
        # Skip update if not enough experiences
        if len(self.experience_buffer) < self.batch_size:
            return None
            
        # Sample batch from experience buffer
        batch = self.experience_buffer.sample(self.batch_size)
        
        # Prepare batch data for GRPO
        states = torch.stack([exp['state'] for exp in batch])
        log_probs_old = torch.tensor([exp['log_prob'].item() for exp in batch])
        rewards = torch.tensor([exp['reward'] for exp in batch])
        actions = torch.tensor([exp['action'] for exp in batch])
        
        # Forward pass through policy to get new log probabilities
        all_log_probs = self.policy_network(states)
        log_probs_new = torch.stack([all_log_probs[i, action] for i, action in enumerate(actions)])
        
        batch_data = {
            "log_probs_old": log_probs_old.numpy(),
            "log_probs_ref": log_probs_old.numpy(),  # Use old probs as reference
            "rewards": rewards.numpy(),
            "group_size": self.batch_size,
            "log_probs_new": log_probs_new  # Pass log_probs_new directly in batch_data
        }
        
        # Train the policy network
        loss, metrics = self.grpo.train_step(self.policy_network, self.optimizer, batch_data, states)
        
        return loss, metrics

def create_agent(agent_type: str, **kwargs) -> BaseAgent:
    """
    Factory function to create an agent based on the agent_type.
    
    Args:
        agent_type (str): The type of agent to create:
            - 'grpo': Discrete action space GRPO agent
            - 'continuous_grpo': Continuous action space GRPO agent
        **kwargs: Additional keyword arguments to pass to the agent constructor.
        
    Returns:
        BaseAgent: An instance of a subclass of BaseAgent.
        
    Raises:
        ValueError: If agent_type is not recognized.
    """
    agent_type = agent_type.lower()
    if agent_type == "grpo":
        return GRPOAgent(**kwargs)
    elif agent_type == "continuous_grpo":
        return ContinuousGRPOAgent(**kwargs)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

class ContinuousGRPOAgent(BaseAgent):
    """
    GRPO agent for continuous action spaces that uses a Gaussian policy.
    
    This agent models continuous actions using a Gaussian distribution,
    where the policy network outputs the mean and log standard deviation.
    """
    
    def __init__(self, 
                 policy_network: nn.Module, 
                 optimizer_class: Any,
                 action_dim: int,
                 learning_rate: float = 0.001, 
                 gamma: float = 0.99, 
                 grpo_params: Dict[str, float] = None,
                 buffer_capacity: int = 10000,
                 batch_size: int = 64,
                 min_std: float = 0.01,
                 action_bounds: Optional[Tuple[float, float]] = None):
        """
        Initialize a continuous GRPO agent.
        
        Args:
            policy_network (nn.Module): Network that outputs mean and log_std for each action dimension
            optimizer_class: PyTorch optimizer class
            action_dim (int): Dimensionality of the action space
            learning_rate (float): Learning rate for optimization
            gamma (float): Discount factor for future rewards
            grpo_params (dict): Parameters for GRPO (epsilon, beta)
            buffer_capacity (int): Maximum size of experience buffer
            batch_size (int): Number of samples for each update
            min_std (float): Minimum standard deviation for numerical stability
            action_bounds (tuple, optional): (min, max) bounds for actions
        """
        self.policy_network = policy_network
        self.optimizer = optimizer_class(self.policy_network.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.grpo = GRPO(**(grpo_params if grpo_params is not None else {}))
        self.memory = []
        self.experience_buffer = ExperienceBuffer(capacity=buffer_capacity)
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.min_std = min_std
        self.action_bounds = action_bounds
    
    def act(self, state: Any) -> np.ndarray:
        """
        Select a continuous action based on the current state.
        
        Args:
            state: The current environment state
            
        Returns:
            np.ndarray: The selected continuous action vector
        """
        state_tensor = torch.tensor(state, dtype=torch.float32)
        # Get action distribution parameters from policy network
        network_output = self.policy_network(state_tensor)
        
        # Split output into mean and log_std
        mean = network_output[:self.action_dim]
        log_std = network_output[self.action_dim:]
        
        # Ensure minimum standard deviation
        std = torch.exp(log_std) + self.min_std
        
        # Sample from the Gaussian distribution
        normal_distribution = torch.distributions.Normal(mean, std)
        action = normal_distribution.sample()
        log_prob = normal_distribution.log_prob(action).sum()
        
        # Apply action bounds if specified
        if self.action_bounds is not None:
            min_action, max_action = self.action_bounds
            action = torch.clamp(action, min_action, max_action)
        
        # Store experience
        self.memory.append({
            'state': state_tensor,
            'action': action,
            'mean': mean,
            'std': std,
            'log_prob': log_prob
        })
        
        return action.detach().numpy()
    
    def store_experience(self, reward: float, done: bool = False) -> None:
        """
        Store the most recent experience with its reward.
        
        Args:
            reward (float): The reward received for the last action
            done (bool): Whether the episode has ended
        """
        if not self.memory:
            return
            
        experience = self.memory[-1]
        experience['reward'] = reward
        experience['done'] = done
        
        # Add to experience buffer for replay
        self.experience_buffer.add(experience.copy())
    
    def update(self, reward: float = None) -> Any:
        """
        Update the policy network using experience replay.
        
        Args:
            reward (float, optional): If provided, stores this reward before updating
            
        Returns:
            tuple or None: (loss, metrics) if update was performed, None otherwise
        """
        # If reward is provided, store the experience first
        if reward is not None:
            self.store_experience(reward)
            
        # Skip update if not enough experiences
        if len(self.experience_buffer) < self.batch_size:
            return None
            
        # Sample batch from experience buffer
        batch = self.experience_buffer.sample(self.batch_size)
        
        # Prepare batch data for GRPO
        states = torch.stack([exp['state'] for exp in batch])
        log_probs_old = torch.tensor([exp['log_prob'].item() for exp in batch])
        rewards = torch.tensor([exp['reward'] for exp in batch])
        actions = torch.stack([exp['action'] for exp in batch])
        
        # Forward pass to get new log_probs
        network_output = self.policy_network(states)
        means = network_output[:, :self.action_dim]
        log_stds = network_output[:, self.action_dim:]
        stds = torch.exp(log_stds) + self.min_std
        
        # Create distributions and compute log probabilities
        normal_distributions = torch.distributions.Normal(means, stds)
        log_probs_new = normal_distributions.log_prob(actions).sum(dim=1)
        
        batch_data = {
            "log_probs_old": log_probs_old.numpy(),
            "log_probs_ref": log_probs_old.numpy(),  # Use old probs as reference
            "rewards": rewards.numpy(),
            "group_size": self.batch_size,
            "log_probs_new": log_probs_new  # Pass log_probs_new directly in batch_data
        }
        
        # Train the policy network
        loss, metrics = self.grpo.train_step(self.policy_network, self.optimizer, batch_data, states)
        
        return loss, metrics