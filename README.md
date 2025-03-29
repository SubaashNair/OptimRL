# 🚀 OptimRL: Group Relative Policy Optimization


OptimRL is a **high-performance reinforcement learning library** that introduces a groundbreaking algorithm, **Group Relative Policy Optimization (GRPO)**. Designed to streamline the training of RL agents, GRPO eliminates the need for a critic network while ensuring robust performance with **group-based advantage estimation** and **KL regularization**. Whether you're building an AI to play games, optimize logistics, or manage resources, OptimRL provides **state-of-the-art efficiency and stability**.

## 🏅 Badges

![PyPI Version](https://img.shields.io/pypi/v/optimrl)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/Library-NumPy-013243?logo=numpy&logoColor=white)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Setuptools](https://img.shields.io/badge/Tool-Setuptools-3776AB?logo=python&logoColor=white)
![Build Status](https://github.com/subaashnair/optimrl/actions/workflows/tests.yml/badge.svg)
![License](https://img.shields.io/github/license/subaashnair/optimrl)
<!-- ![Coverage](https://img.shields.io/codecov/c/github/subaashnair/optimrl) -->


## 🌟 Features

### Why Choose OptimRL?

1. **🚫 Critic-Free Learning**  
   Traditional RL methods require training both an actor and a critic network. GRPO eliminates this dual-network requirement, cutting **model complexity by 50%** while retaining top-tier performance.  

2. **👥 Group-Based Advantage Estimation**  
   GRPO introduces a novel way to normalize rewards within groups of experiences. This ensures:
   - **Stable training** across diverse reward scales.
   - Adaptive behavior for varying tasks and environments.

3. **📏 KL Regularization**  
   Prevent **policy collapse** with GRPO's built-in KL divergence regularization, ensuring:
   - **Smoothed updates** for policies.
   - Reliable and stable learning in any domain.

4. **⚡ Vectorized NumPy Operations with PyTorch Tensor Integration**  
   OptimRL leverages **NumPy's vectorized operations** and **PyTorch's tensor computations** with GPU acceleration for maximum performance. This hybrid implementation provides:
   - **10-100x speedups** over pure Python through optimized array programming
   - Seamless CPU/GPU execution via PyTorch backend
   - Native integration with deep learning workflows
   - Full automatic differentiation support

5. **🔄 Experience Replay Buffer**  
   Improve sample efficiency with built-in experience replay:
   - Learn from past experiences multiple times
   - Reduce correlation between consecutive samples
   - Configurable buffer capacity and batch sizes

6. **🔄 Continuous Action Space Support**  
   Train agents in environments with continuous control:
   - Gaussian policy implementation for continuous actions
   - Configurable action bounds
   - Adaptive standard deviation for exploration

---

## 🛠️ Installation

### For End Users
Simply install from PyPI:
```bash
pip install optimrl
```

### For Developers
Clone the repository and set up a development environment:
```bash
git clone https://github.com/subaashnair/optimrl.git
cd optimrl
pip install -e '.[dev]'
```

---

## ⚡ Quick Start

### Discrete Action Space Example (CartPole)

```python
import torch
import torch.nn as nn
import torch.optim as optim
import gym
from optimrl import create_agent

# Define a simple policy network
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

# Create environment and network
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
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
    batch_size=32
)

# Training loop
state, _ = env.reset()
for step in range(1000):
    action = agent.act(state)
    next_state, reward, done, truncated, _ = env.step(action)
    agent.store_experience(reward, done)
    
    if done or truncated:
        state, _ = env.reset()
        agent.update()  # Update policy after episode ends
    else:
        state = next_state
```

### Complete CartPole Implementation

For a complete implementation of CartPole with OptimRL, check out our examples in the `simple_test` directory:

- `cartpole_simple.py`: Basic implementation with GRPO
- `cartpole_improved.py`: Improved implementation with tuned parameters
- `cartpole_final.py`: Final implementation with optimized performance
- `cartpole_tuned.py`: Enhanced implementation with advanced features
- `cartpole_simple_pg.py`: Vanilla Policy Gradient implementation for comparison

The vanilla policy gradient implementation (`cartpole_simple_pg.py`) achieves excellent performance on CartPole-v1, reaching the maximum reward of 500 consistently. It serves as a useful baseline for comparing against the GRPO implementations.

### Continuous Action Space Example (Pendulum)

```python
import torch
import torch.nn as nn
import torch.optim as optim
import gym
from optimrl import create_agent

# Define a continuous policy network
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
        return self.output_layer(x)

# Create environment and network
env = gym.make('Pendulum-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bounds = (env.action_space.low[0], env.action_space.high[0])
policy = ContinuousPolicyNetwork(state_dim, action_dim)

# Create Continuous GRPO agent
agent = create_agent(
    "continuous_grpo",
    policy_network=policy,
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
state, _ = env.reset()
for step in range(1000):
    action = agent.act(state)
    next_state, reward, done, truncated, _ = env.step(action)
    agent.store_experience(reward, done)
    
    if done or truncated:
        state, _ = env.reset()
        agent.update()  # Update policy after episode ends
    else:
        state = next_state
```

## 📊 Performance Comparison

Our simple policy gradient implementation consistently solves the CartPole-v1 environment in under 1000 episodes, achieving the maximum reward of 500. The GRPO implementations offer competitive performance with additional benefits:

- **Lower variance**: More stable learning across different random seeds
- **Improved sample efficiency**: Learns from fewer interactions with the environment
- **Better regularization**: Prevents policy collapse during training

## Kaggle Notebook

You can view the "OptimRL Trading Experiment" notebook on Kaggle:
[![OptimRL Trading Experiment](https://img.shields.io/badge/Kaggle-OptimRL_Trading_Experiment-orange)](https://www.kaggle.com/code/noir1112/optimrl-trading-experiment/edit)

Alternatively, you can open the notebook locally as an `.ipynb` file:
[Open the OptimRL Trading Experiment Notebook (.ipynb)](./notebooks/OptimRL_Trading_Experiment.ipynb)

---

## 🤝 Contributing

We're excited to have you onboard! Here's how you can help improve **OptimRL**:
1. **Fork the repo.**  
2. **Create a feature branch**:  
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit your changes**:  
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push to the branch**:  
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open a Pull Request**.  

Before submitting, make sure you run all tests:
```bash
pytest tests/
```

---

## 📜 License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

---

## 📚 Citation

If you use OptimRL in your research, please cite:

```bibtex
@software{optimrl2024,
  title={OptimRL: Group Relative Policy Optimization},
  author={Subashan Nair},
  year={2024},
  url={https://github.com/subaashnair/optimrl}
}
```

---




