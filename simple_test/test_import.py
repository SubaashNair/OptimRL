import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from optimrl import GRPO, GRPOAgent, ContinuousGRPOAgent, create_agent

class SimpleNetwork(nn.Module):
    def __init__(self, input_dim=4, output_dim=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.LogSoftmax(dim=-1)
        )
    
    def forward(self, x):
        return self.network(x)

# Test GRPO instantiation
grpo = GRPO(epsilon=0.2, beta=0.1)
print("GRPO initialized successfully!")

# Test GRPOAgent instantiation
network = SimpleNetwork()
agent = create_agent(
    "grpo",
    policy_network=network,
    optimizer_class=optim.Adam,
    learning_rate=0.001
)
print("GRPOAgent created successfully!")

# Test basic agent functionality
state = np.array([0.1, 0.2, 0.3, 0.4])
action = agent.act(state)
print(f"Agent selected action: {action}")

# Test continuous agent
cont_network = SimpleNetwork(output_dim=4)  # 2 actions, each with mean and std
cont_agent = create_agent(
    "continuous_grpo",
    policy_network=cont_network,
    optimizer_class=optim.Adam,
    action_dim=2,
    learning_rate=0.001
)
print("ContinuousGRPOAgent created successfully!")

# Test continuous agent action
cont_action = cont_agent.act(state)
print(f"Continuous agent selected action: {cont_action}")

print("All tests passed!") 