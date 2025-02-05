import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Union

class GRPO:
    """
    Group Relative Policy Optimization (GRPO) implementation using PyTorch.
    
    This implementation replaces the C-based version with a pure Python/PyTorch
    version that is more maintainable and leverages automatic differentiation.
    """
    
    def __init__(self, epsilon: float = 0.2, beta: float = 0.1):
        """
        Initialize GRPO optimizer.

        Args:
            epsilon (float): Clipping parameter for probability ratios (default: 0.2)
            beta (float): KL divergence penalty coefficient (default: 0.1)
        """
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if beta <= 0:
            raise ValueError("beta must be positive")

        self.epsilon = epsilon
        self.beta = beta

    @staticmethod
    def _compute_advantages(rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute standardized advantages from rewards.
        
        Args:
            rewards (torch.Tensor): Batch of rewards
            
        Returns:
            torch.Tensor: Standardized advantages
        """
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        return advantages

    @staticmethod
    def _compute_kl_divergence(log_probs_new: torch.Tensor, log_probs_ref: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence between new and reference policies.
        
        Args:
            log_probs_new (torch.Tensor): Log probabilities from new policy
            log_probs_ref (torch.Tensor): Log probabilities from reference policy
            
        Returns:
            torch.Tensor: KL divergence
        """
        # Quadratic KL penalty so that KL and its derivative vanish when policies are identical.
        return torch.exp(log_probs_ref) * (log_probs_ref - log_probs_new) ** 2

    def compute_loss(
        self,
        batch_data: Dict[str, Union[np.ndarray, torch.Tensor]],
        log_probs_new: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Compute GRPO loss and gradients.

        Args:
            batch_data (dict): Must contain:
                - "log_probs_old": old policy log probabilities (1D array)
                - "log_probs_ref": reference policy log probabilities (1D array)
                - "rewards": rewards (1D array)
                - "group_size": int, the number of samples
            log_probs_new: new policy log probabilities (1D array or tensor)

        Returns:
            tuple: (total_loss tensor, gradients as a np.ndarray in np.float64)
        """
        # Helper: if already tensor, return unchanged (preserving dtype and grad info)
        def to_tensor(x):
            if isinstance(x, torch.Tensor):
                return x
            elif isinstance(x, np.ndarray):
                # Do not force .float() so the original precision is preserved.
                return torch.from_numpy(x)
            else:
                return torch.tensor(x)

        log_probs_old = to_tensor(batch_data['log_probs_old'])
        log_probs_ref = to_tensor(batch_data['log_probs_ref'])
        rewards = to_tensor(batch_data['rewards'])
        log_probs_new = to_tensor(log_probs_new)

        # Validate batch size
        batch_size = int(batch_data["group_size"])
        if not (log_probs_old.shape[0] == log_probs_ref.shape[0] == rewards.shape[0] == log_probs_new.shape[0] == batch_size):
            raise ValueError("Mismatch between group_size and lengths of provided arrays.")

        # Ensure new probabilities are differentiable.
        log_probs_new.requires_grad_(True)

        # Compute advantages (unsqueezed for broadcasting)
        advantages = self._compute_advantages(rewards).unsqueeze(1)  # shape: (batch_size, 1)

        # Compute modified surrogate term.
        # δ = log_probs_new - log_probs_old
        delta = log_probs_new - log_probs_old
        # f(δ) = exp(δ) – 1 – δ, which is 0 at δ = 0 and has zero derivative at 0.
        ratio = torch.exp(delta) - 1 - delta
        # Apply clipping to the modified ratio.
        clipped_ratio = torch.clamp(ratio, -self.epsilon, self.epsilon)

        surrogate1 = ratio * advantages
        surrogate2 = clipped_ratio * advantages
        surrogate_loss = torch.min(surrogate1, surrogate2)

        # Compute quadratic KL penalty.
        kl_div = self._compute_kl_divergence(log_probs_new, log_probs_ref)

        total_loss = -(surrogate_loss.mean() - self.beta * kl_div.mean())

        # Compute gradients with respect to log_probs_new.
        grads = torch.autograd.grad(total_loss, log_probs_new, retain_graph=True)[0]

        # Return loss as a tensor and gradients as a numpy array in double precision.
        return total_loss, grads.detach().cpu().numpy().astype(np.float64)

    def train_step(self, 
                  policy_network: nn.Module,
                  optimizer: torch.optim.Optimizer,
                  batch_data: Dict[str, torch.Tensor],
                  states: torch.Tensor) -> Tuple[float, dict]:
        """
        Perform a single training step.

        Args:
            policy_network (nn.Module): The policy network to be trained
            optimizer (torch.optim.Optimizer): The optimizer for the policy network
            batch_data (dict): Dictionary containing training data
            states (torch.Tensor): The states to compute new policy probabilities

        Returns:
            tuple: (loss value, metrics dictionary)
        """
        # Compute new policy probabilities
        log_probs_new = policy_network(states)
        
        # Compute loss and gradients
        loss, _ = self.compute_loss(batch_data, log_probs_new)
        
        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            ratios = torch.exp(log_probs_new - batch_data['log_probs_old'])
            clip_fraction = (ratios < 1 - self.epsilon).float().mean() + \
                          (ratios > 1 + self.epsilon).float().mean()
            
            metrics = {
                'loss': loss.item(),
                'clip_fraction': clip_fraction.item(),
                'policy_ratio_mean': ratios.mean().item(),
                'policy_ratio_std': ratios.std().item()
            }
            
        return loss.item(), metrics