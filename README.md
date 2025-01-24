# 🚀 OptimRL: Group Relative Policy Optimization


OptimRL is a **high-performance reinforcement learning library** that introduces a groundbreaking algorithm, **Group Relative Policy Optimization (GRPO)**. Designed to streamline the training of RL agents, GRPO eliminates the need for a critic network while ensuring robust performance with **group-based advantage estimation** and **KL regularization**. Whether you're building an AI to play games, optimize logistics, or manage resources, OptimRL provides **state-of-the-art efficiency and stability**.

## 🏅 Badges

![PyPI Version](https://img.shields.io/pypi/v/optimrl)
![PyPI Downloads](https://img.shields.io/pypi/dm/optimrl)
![Python Version](https://img.shields.io/pypi/pyversions/optimrl)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![C](https://img.shields.io/badge/C-99-00599C?logo=c&logoColor=white)
![NumPy](https://img.shields.io/badge/Library-NumPy-013243?logo=numpy&logoColor=white)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Setuptools](https://img.shields.io/badge/Tool-Setuptools-3776AB?logo=python&logoColor=white)
![Build Status](https://github.com/subaashnair/optimrl/actions/workflows/tests.yml/badge.svg)
![CI](https://github.com/subaashnair/optimrl/workflows/CI/badge.svg)
![Coverage](https://img.shields.io/codecov/c/github/subaashnair/optimrl)
![License](https://img.shields.io/github/license/subaashnair/optimrl)

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

4. **⚡ High-Performance C Implementation with Python Interface**  
   At its core, OptimRL is written in **C** for maximum computational efficiency and wrapped in **Python** for ease of use. This gives you the **best of both worlds**: blazing-fast performance and developer-friendly APIs.

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

Here’s a **minimal working example** to get started with OptimRL:

```python
import torch
import optimrl

# Initialize the GRPO optimizer
grpo = optimrl.GRPO(epsilon=0.2, beta=0.1)

# Prepare batch data (example)
batch_data = {
    'log_probs_old': current_policy_log_probs,
    'log_probs_ref': reference_policy_log_probs,
    'rewards': episode_rewards,
    'group_size': len(episode_rewards)
}

# Compute policy loss
log_probs_new = new_policy_log_probs
loss, gradients = grpo.compute_loss(batch_data, log_probs_new)

# Apply gradients to update the policy
optimizer.zero_grad()
policy_loss = torch.tensor(loss, requires_grad=True)
policy_loss.backward()
optimizer.step()
```

---

## 🔍 Advanced Usage

Integrate OptimRL seamlessly into your **PyTorch pipelines** or custom training loops. Below is a **complete example** showcasing GRPO in action:

```python
import torch
import optimrl

class PolicyNetwork(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, output_dim),
            torch.nn.LogSoftmax(dim=-1)
        )
    
    def forward(self, x):
        return self.network(x)

# Initialize components
policy = PolicyNetwork(input_dim=4, output_dim=2)
reference_policy = PolicyNetwork(input_dim=4, output_dim=2)
optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
grpo = optimrl.GRPO(epsilon=0.2, beta=0.1)

# Training loop
for episode in range(1000):  # Replace with your num_episodes
    states, actions, rewards = collect_episode()  # Replace with your data
    
    # Compute log probabilities
    with torch.no_grad():
        log_probs_old = policy(states)
        log_probs_ref = reference_policy(states)
    
    batch_data = {
        'log_probs_old': log_probs_old.numpy(),
        'log_probs_ref': log_probs_ref.numpy(),
        'rewards': rewards,
        'group_size': len(rewards)
    }
    
    # Policy update
    log_probs_new = policy(states)
    loss, gradients = grpo.compute_loss(batch_data, log_probs_new.numpy())
    
    # Backpropagation
    optimizer.zero_grad()
    policy_loss = torch.tensor(loss, requires_grad=True)
    policy_loss.backward()
    optimizer.step()
```

---

## 🤝 Contributing

We’re excited to have you onboard! Here’s how you can help improve **OptimRL**:
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
  author={Your Name},
  year={2024},
  url={https://github.com/subaashnair/optimrl}
}
```

---


