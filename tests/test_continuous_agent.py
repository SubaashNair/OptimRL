#!/usr/bin/env python
import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from optimrl import ContinuousGRPOAgent

class SimpleModel(nn.Module):
    """A simple model for testing ContinuousGRPOAgent."""
    def __init__(self, input_dim=4, action_dim=2):
        super().__init__()
        self.layer = nn.Linear(input_dim, action_dim * 2)  # mean and log_std

    def forward(self, x):
        return self.layer(x)

@pytest.fixture
def continuous_agent():
    """Create a ContinuousGRPOAgent for testing."""
    model = SimpleModel()
    agent = ContinuousGRPOAgent(
        policy_network=model,
        optimizer_class=optim.Adam,
        action_dim=2,
        learning_rate=0.001,
        batch_size=4
    )
    return agent

def test_agent_initialization(continuous_agent):
    """Test the initialization of the ContinuousGRPOAgent."""
    assert continuous_agent.action_dim == 2
    assert continuous_agent.min_std == 0.01
    assert continuous_agent.action_bounds is None
    assert len(continuous_agent.memory) == 0
    assert len(continuous_agent.experience_buffer) == 0

def test_agent_act():
    """Test the act method of the ContinuousGRPOAgent."""
    model = SimpleModel(input_dim=3, action_dim=1)
    agent = ContinuousGRPOAgent(
        policy_network=model,
        optimizer_class=optim.Adam,
        action_dim=1,
        learning_rate=0.001,
        batch_size=4,
        action_bounds=(-2.0, 2.0)  # Test with bounds
    )
    
    # Test with a simple state
    state = np.array([0.1, 0.2, 0.3])
    action = agent.act(state)
    
    # Check action shape and bounds
    assert isinstance(action, np.ndarray)
    assert action.shape == (1,)
    assert action[0] >= -2.0 and action[0] <= 2.0
    
    # Check that memory was updated
    assert len(agent.memory) == 1
    assert 'state' in agent.memory[0]
    assert 'action' in agent.memory[0]
    assert 'log_prob' in agent.memory[0]

def test_store_experience(continuous_agent):
    """Test storing experience."""
    # First create an action to populate memory
    state = np.array([0.1, 0.2, 0.3, 0.4])
    continuous_agent.act(state)
    
    # Store the experience
    reward = 1.0
    continuous_agent.store_experience(reward, done=False)
    
    # Check that experience was stored in buffer
    assert len(continuous_agent.experience_buffer) == 1
    exp = next(iter(continuous_agent.experience_buffer.buffer))
    assert 'reward' in exp
    assert exp['reward'] == 1.0
    assert 'done' in exp
    assert exp['done'] is False

def test_update_with_insufficient_data(continuous_agent):
    """Test update method with insufficient data."""
    result = continuous_agent.update()
    assert result is None

def test_update_with_data():
    """Test update method with sufficient data."""
    # Create an agent with smaller batch size for testing
    model = SimpleModel()
    agent = ContinuousGRPOAgent(
        policy_network=model,
        optimizer_class=optim.Adam,
        action_dim=2,
        learning_rate=0.001,
        batch_size=2  # Small batch size for testing
    )
    
    # Generate some experiences
    for i in range(3):  # More than batch_size
        state = np.array([0.1, 0.2, 0.3, 0.4])
        agent.act(state)
        agent.store_experience(reward=1.0, done=False)
    
    # Should be able to perform an update now
    result = agent.update()
    
    # Should return loss and metrics
    assert result is not None
    loss, metrics = result
    assert isinstance(loss, float)
    assert isinstance(metrics, dict)
    assert 'loss' in metrics
