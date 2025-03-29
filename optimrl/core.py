import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Union

class GRPO:
    """
    Group Relative Policy Optimization (GRPO) implementation using PyTorch.
    
    GRPO is a policy optimization algorithm that eliminates the need for a critic network
    while maintaining performance through group-based advantage estimation and KL regularization.
    
    Key features:
    - Critic-free learning: Reduces model complexity by 50% compared to actor-critic methods
    - Group-based advantage estimation: Stabilizes training across diverse reward scales
    - KL regularization: Prevents policy collapse with smooth updates
    
    This implementation uses a hybrid approach combining NumPy's vectorized operations
    with PyTorch's tensor computations and automatic differentiation.
    
    Example:
        ```python
        import torch
        import torch.nn as nn
        from optimrl import GRPO
        
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
        
        # Initialize GRPO and model
        policy = PolicyNetwork(input_dim=4, output_dim=2)
        optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
        grpo = GRPO(epsilon=0.2, beta=0.1)
        
        # In your training loop:
        states_batch = torch.tensor([...])  # Your environment states
        batch_data = {
            "log_probs_old": [...],  # Log probs from old policy
            "log_probs_ref": [...],  # Reference policy log probs
            "rewards": [...],        # Rewards
            "group_size": len(states_batch)
        }
        
        loss, metrics = grpo.train_step(policy, optimizer, batch_data, states_batch)
        ```
    """
    
    def __init__(self, epsilon: float = 0.2, beta: float = 0.1):
        """
        Initialize GRPO optimizer.

        Args:
            epsilon (float): Clipping parameter for probability ratios (default: 0.2)
                Controls how far the new policy can deviate from the old policy.
                Lower values enforce more conservative updates.
            beta (float): KL divergence penalty coefficient (default: 0.1)
                Controls how strongly to penalize deviations from the reference policy.
                Higher values result in more conservative updates.
        
        Raises:
            ValueError: If epsilon or beta have invalid values (must be positive)
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
        
        This method standardizes rewards by subtracting the mean and dividing by
        the standard deviation to create advantage estimates that are scale-invariant.
        This normalization makes the algorithm robust to varying reward scales.
        
        Args:
            rewards (torch.Tensor): Batch of rewards with shape (batch_size,)
            
        Returns:
            torch.Tensor: Standardized advantages with shape (batch_size,)
            
        Note:
            A small epsilon (1e-8) is added to the standard deviation to prevent
            division by zero in case all rewards in the batch are identical.
        """
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        return advantages

    @staticmethod
    def _compute_kl_divergence(log_probs_new: torch.Tensor, log_probs_ref: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence between new and reference policies.
        
        Uses a quadratic approximation of KL divergence that vanishes at zero
        difference, creating smoother policy updates. This implementation has 
        favorable properties for policy optimization:
        
        1. Both KL and its derivative are zero when policies are identical
        2. Provides a stronger penalty for larger deviations
        
        Args:
            log_probs_new (torch.Tensor): Log probabilities from new policy, shape (batch_size,)
            log_probs_ref (torch.Tensor): Log probabilities from reference policy, shape (batch_size,)
            
        Returns:
            torch.Tensor: KL divergence estimate with shape (batch_size,)
        """
        # Quadratic KL penalty so that KL and its derivative vanish when policies are identical.
        return torch.exp(log_probs_ref) * (log_probs_ref - log_probs_new) ** 2

    def compute_loss(
        self,
        batch_data: Dict[str, Union[np.ndarray, torch.Tensor]],
        log_probs_new: Union[np.ndarray, torch.Tensor] = None
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Compute GRPO loss and gradients for policy optimization.
        
        This is the core optimization function that combines clipped surrogate loss
        with KL regularization to create smoother, more stable policy updates.
        
        The implementation handles both NumPy arrays and PyTorch tensors as inputs,
        converting everything to tensors for computation and returning gradients
        as NumPy arrays for compatibility.
        
        Args:
            batch_data (dict): Must contain:
                - "log_probs_old" (array/tensor): Old policy log probabilities with shape (batch_size,)
                - "log_probs_ref" (array/tensor): Reference policy log probabilities with shape (batch_size,)
                - "rewards" (array/tensor): Rewards with shape (batch_size,)
                - "group_size" (int): The number of samples in the batch
                - "log_probs_new" (optional): If provided, these will be used instead of the log_probs_new parameter
            log_probs_new (array/tensor, optional): New policy log probabilities with shape (batch_size,)
                If not provided, it will be taken from batch_data["log_probs_new"]
        
        Returns:
            tuple: 
                - total_loss (torch.Tensor): The computed loss value, minimizing this improves policy
                - gradients (np.ndarray): Gradients with respect to log_probs_new in np.float64 precision
                
        Raises:
            ValueError: If there's a mismatch between group_size and length of provided arrays
            KeyError: If any required keys are missing from batch_data
            
        Note:
            The loss function implements a modified clipped surrogate objective that is more
            stable than standard policy gradient approaches. KL divergence regularization
            further ensures policy updates don't deviate too far from the reference policy.
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
        
        # Use log_probs_new from batch_data if provided, otherwise use parameter
        if log_probs_new is None:
            if 'log_probs_new' not in batch_data:
                raise ValueError("log_probs_new must be provided either as parameter or in batch_data")
            log_probs_new = to_tensor(batch_data['log_probs_new'])
        else:
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
        
        # Instead of modifying the ratio with f(δ) = exp(δ) – 1 – δ,
        # use the standard policy ratio for better gradient flow
        ratio = torch.exp(delta)
        
        # Apply clipping to the ratio
        clipped_ratio = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon)

        surrogate1 = ratio * advantages
        surrogate2 = clipped_ratio * advantages
        surrogate_loss = torch.min(surrogate1, surrogate2).mean()

        # Compute KL penalty with scaling to balance with surrogate loss
        kl_div = self._compute_kl_divergence(log_probs_new, log_probs_ref).mean()
        
        # The loss is negative because we want to maximize the surrogate objective
        # while minimizing the KL divergence
        total_loss = -surrogate_loss + self.beta * kl_div

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
        Perform a single training step on the policy network.
        
        This method encapsulates a complete policy optimization step:
        1. Forward pass through the policy network to get new log probabilities
        2. Compute the GRPO loss using the provided batch data
        3. Perform backpropagation and update the network parameters
        4. Compute and return relevant training metrics
        
        Args:
            policy_network (nn.Module): The policy network to be trained
                Should output log probabilities of actions
            optimizer (torch.optim.Optimizer): The optimizer for the policy network
                Typically Adam or SGD
            batch_data (dict): Dictionary containing training data with keys:
                - "log_probs_old": Old policy log probabilities
                - "log_probs_ref": Reference policy log probabilities  
                - "rewards": Rewards received
                - "group_size": Number of samples in the batch
                - "log_probs_new" (optional): If provided, will use these instead of computing from policy_network
            states (torch.Tensor): The input states to compute new policy probabilities
                Shape should be (batch_size, state_dim)

        Returns:
            tuple: 
                - loss (float): Scalar loss value after optimization
                - metrics (dict): Dictionary containing training metrics:
                    - 'loss': Optimization loss value
                    - 'clip_fraction': Fraction of policy ratios that were clipped
                    - 'policy_ratio_mean': Mean ratio between new and old policies
                    - 'policy_ratio_std': Standard deviation of policy ratios
                    
        Note:
            This method handles the entire optimization step, so you don't need to
            manually call backward() or optimizer.step(). Metrics provide insights
            into the training dynamics and can be useful for debugging.
        """
        # Compute new policy probabilities if not provided in batch_data
        if "log_probs_new" in batch_data:
            log_probs_new = batch_data["log_probs_new"]
        else:
            log_probs_new = policy_network(states)
        
        # Compute loss and gradients
        loss, _ = self.compute_loss(batch_data, log_probs_new)
        
        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            ratios = torch.exp(log_probs_new - torch.tensor(batch_data['log_probs_old']))
            clip_fraction = (ratios < 1 - self.epsilon).float().mean() + \
                          (ratios > 1 + self.epsilon).float().mean()
            
            metrics = {
                'loss': loss.item(),
                'clip_fraction': clip_fraction.item(),
                'policy_ratio_mean': ratios.mean().item(),
                'policy_ratio_std': ratios.std().item()
            }
            
        return loss.item(), metrics