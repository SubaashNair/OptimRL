import pytest
import numpy as np
from optimrl.core import GRPO

@pytest.fixture
def sample_batch():
    """Create a sample batch of data for testing."""
    group_size = 5
    return {
        'log_probs_old': np.random.randn(group_size).astype(np.float64),
        'log_probs_ref': np.random.randn(group_size).astype(np.float64),
        'rewards': np.random.randn(group_size).astype(np.float64),
        'group_size': group_size
    }

def test_grpo_initialization():
    """Test GRPO initialization with different parameters."""
    # Test default parameters
    grpo = GRPO()
    assert grpo.epsilon == 0.2
    assert grpo.beta == 0.1

    # Test custom parameters
    grpo = GRPO(epsilon=0.3, beta=0.2)
    assert grpo.epsilon == 0.3
    assert grpo.beta == 0.2

    # Test invalid parameters
    with pytest.raises(ValueError):
        GRPO(epsilon=-0.1)  # Epsilon should be positive
    with pytest.raises(ValueError):
        GRPO(beta=-0.1)    # Beta should be positive

def test_compute_loss_basic(sample_batch):
    """Test basic loss computation functionality."""
    grpo = GRPO()
    log_probs_new = np.random.randn(sample_batch['group_size']).astype(np.float64)
    
    loss, grad = grpo.compute_loss(sample_batch, log_probs_new)
    
    # Check output types and shapes
    assert isinstance(loss, float)
    assert isinstance(grad, np.ndarray)
    assert grad.shape == log_probs_new.shape
    assert grad.dtype == np.float64
    
    # Check for valid values
    assert not np.isnan(loss)
    assert not np.any(np.isnan(grad))
    assert not np.any(np.isinf(grad))

def test_compute_loss_identical_policies(sample_batch):
    """Test loss computation when policies are identical."""
    grpo = GRPO()
    log_probs_new = sample_batch['log_probs_old'].copy()
    
    loss, grad = grpo.compute_loss(sample_batch, log_probs_new)
    
    # When policies are identical, gradients should be small
    assert np.all(np.abs(grad) < 1e-5)
    
    # Loss should be close to zero when policies are identical and rewards are standardized
    standardized_batch = sample_batch.copy()
    standardized_batch['rewards'] = (sample_batch['rewards'] - np.mean(sample_batch['rewards'])) / np.std(sample_batch['rewards'])
    loss_standardized, _ = grpo.compute_loss(standardized_batch, log_probs_new)
    assert abs(loss_standardized) < 1e-5

def test_input_validation():
    """Test input validation for various error cases."""
    grpo = GRPO()
    
    # Test invalid group size
    invalid_batch = {
        'log_probs_old': np.random.randn(3).astype(np.float64),
        'log_probs_ref': np.random.randn(4).astype(np.float64),
        'rewards': np.random.randn(3).astype(np.float64),
        'group_size': 3
    }
    
    with pytest.raises(ValueError):
        grpo.compute_loss(invalid_batch, np.random.randn(4).astype(np.float64))
    
    # Test invalid data types
    float32_batch = {
        'log_probs_old': np.random.randn(3).astype(np.float32),
        'log_probs_ref': np.random.randn(3).astype(np.float32),
        'rewards': np.random.randn(3).astype(np.float32),
        'group_size': 3
    }
    
    # Should handle float32 inputs by converting them
    loss, grad = grpo.compute_loss(float32_batch, np.random.randn(3).astype(np.float32))
    assert isinstance(loss, float)
    assert grad.dtype == np.float64

    # Test missing keys
    incomplete_batch = {
        'log_probs_old': np.random.randn(3).astype(np.float64),
        'group_size': 3
    }
    with pytest.raises(KeyError):
        grpo.compute_loss(incomplete_batch, np.random.randn(3).astype(np.float64))

def test_reward_scaling(sample_batch):
    """Test that the algorithm is invariant to reward scaling."""
    grpo = GRPO()
    log_probs_new = np.random.randn(sample_batch['group_size']).astype(np.float64)
    
    # Compute loss with original rewards
    loss1, grad1 = grpo.compute_loss(sample_batch, log_probs_new)
    
    # Scale rewards and compute loss again
    scaled_batch = sample_batch.copy()
    scaled_batch['rewards'] = sample_batch['rewards'] * 10.0
    
    loss2, grad2 = grpo.compute_loss(scaled_batch, log_probs_new)
    
    # Gradients should be similar despite reward scaling
    assert np.allclose(grad1, grad2, rtol=1e-5, atol=1e-8)

def test_numerical_stability():
    """Test numerical stability with extreme values."""
    grpo = GRPO()
    group_size = 5
    
    # Test with very large rewards
    large_batch = {
        'log_probs_old': np.random.randn(group_size).astype(np.float64),
        'log_probs_ref': np.random.randn(group_size).astype(np.float64),
        'rewards': np.random.randn(group_size).astype(np.float64) * 1e6,
        'group_size': group_size
    }
    
    log_probs_new = np.random.randn(group_size).astype(np.float64)
    loss, grad = grpo.compute_loss(large_batch, log_probs_new)
    
    assert not np.isnan(loss)
    assert not np.any(np.isnan(grad))
    assert not np.any(np.isinf(grad))
    
    # Test with very small rewards
    small_batch = large_batch.copy()
    small_batch['rewards'] = np.random.randn(group_size).astype(np.float64) * 1e-6
    
    loss, grad = grpo.compute_loss(small_batch, log_probs_new)
    
    assert not np.isnan(loss)
    assert not np.any(np.isnan(grad))
    assert not np.any(np.isinf(grad))

def test_kl_penalty():
    """Test the effect of KL divergence penalty."""
    # Initialize with high beta (strong KL penalty)
    grpo_high_kl = GRPO(beta=1.0)
    grpo_low_kl = GRPO(beta=0.01)
    
    group_size = 5
    batch = {
        'log_probs_old': np.zeros(group_size).astype(np.float64),
        'log_probs_ref': np.zeros(group_size).astype(np.float64),
        'rewards': np.ones(group_size).astype(np.float64),
        'group_size': group_size
    }
    
    # Try to deviate significantly from reference policy
    log_probs_new = np.ones(group_size).astype(np.float64)
    
    # Compute losses with different KL penalties
    loss_high_kl, grad_high_kl = grpo_high_kl.compute_loss(batch, log_probs_new)
    loss_low_kl, grad_low_kl = grpo_low_kl.compute_loss(batch, log_probs_new)
    
    # Higher KL penalty should result in larger loss
    assert abs(loss_high_kl) > abs(loss_low_kl)
    # Higher KL penalty should result in larger gradients
    assert np.mean(np.abs(grad_high_kl)) > np.mean(np.abs(grad_low_kl))

# def test_clipping_behavior():
#     """Test the policy clipping mechanism."""
#     grpo = GRPO(epsilon=0.2)  # 20% clipping
#     group_size = 5
    
#     # Create a batch where the new policy deviates significantly from the old policy
#     batch = {
#         'log_probs_old': np.zeros(group_size).astype(np.float64),
#         'log_probs_ref': np.zeros(group_size).astype(np.float64),
#         'rewards': np.ones(group_size).astype(np.float64),
#         'group_size': group_size
#     }
    
#     # Test with policy updates of different magnitudes
#     small_update = np.ones(group_size).astype(np.float64) * 0.1  # Within clipping range
#     large_update = np.ones(group_size).astype(np.float64) * 0.5  # Outside clipping range
    
#     _, grad_small = grpo.compute_loss(batch, small_update)
#     _, grad_large = grpo.compute_loss(batch, large_update)
    
#     # Gradients should be larger for the small update (unclipped) than the large update (clipped)
#     assert np.mean(np.abs(grad_small)) > np.mean(np.abs(grad_large))

def test_clipping_behavior():
    """Test the policy clipping mechanism."""
    grpo = GRPO(epsilon=0.2)  # 20% clipping
    group_size = 5
    
    # Create a batch with positive advantages to test clipping
    batch = {
        'log_probs_old': np.zeros(group_size).astype(np.float64),
        'log_probs_ref': np.zeros(group_size).astype(np.float64),
        'rewards': np.ones(group_size).astype(np.float64),
        'group_size': group_size
    }
    
    # Test range of policy updates
    update_sizes = np.linspace(0.1, 0.5, 5)
    grads = []
    
    for size in update_sizes:
        update = np.ones(group_size).astype(np.float64) * size
        _, grad = grpo.compute_loss(batch, update)
        grads.append(np.mean(np.abs(grad)))
    
    # For updates beyond clipping threshold, gradient magnitude should plateau
    grads = np.array(grads)
    grad_changes = np.diff(grads)
    
    # Gradient changes should be smaller for larger updates (indicating clipping)
    assert np.mean(np.abs(grad_changes[:2])) > np.mean(np.abs(grad_changes[2:]))

def test_batch_consistency():
    """Test that the algorithm behaves consistently across different batch sizes."""
    grpo = GRPO()
    
    # Create controlled data with identical patterns
    base_rewards = np.array([1.0, -1.0, 0.5, -0.5, 0.0])
    base_log_probs = np.array([-0.5, -0.5, -0.5, -0.5, -0.5])
    
    # Create batches
    batch_small = {
        'log_probs_old': base_log_probs.astype(np.float64),
        'log_probs_ref': base_log_probs.astype(np.float64),
        'rewards': base_rewards.astype(np.float64),
        'group_size': 5
    }
    
    batch_large = {
        'log_probs_old': np.tile(base_log_probs, 2).astype(np.float64),
        'log_probs_ref': np.tile(base_log_probs, 2).astype(np.float64),
        'rewards': np.tile(base_rewards, 2).astype(np.float64),
        'group_size': 10
    }
    
    # Use consistent policy updates
    log_probs_small = np.full(5, -0.3, dtype=np.float64)
    log_probs_large = np.full(10, -0.3, dtype=np.float64)
    
    loss_small, grad_small = grpo.compute_loss(batch_small, log_probs_small)
    loss_large, grad_large = grpo.compute_loss(batch_large, log_probs_large)
    
    # Print detailed diagnostic information
    print("\nBatch Consistency Test Diagnostics:")
    print(f"Small batch - Loss: {loss_small:.6f}")
    print(f"Large batch - Loss: {loss_large:.6f}")
    print(f"Small batch gradients: {grad_small}")
    print(f"Large batch gradients: {grad_large}")
    
    # Test loss consistency
    loss_diff = abs(loss_small - loss_large)
    print(f"Loss difference: {loss_diff:.6f}")
    assert loss_diff < 1e-5, f"Loss difference {loss_diff} should be < 1e-5"
    
    # Compute and test gradient scaling
    # grad_small_mean = np.mean(np.abs(grad_small))
    # grad_large_mean = np.mean(np.abs(grad_large))
    # grad_ratio = grad_small_mean / grad_large_mean
    # print(f"Gradient ratio (small/large): {grad_ratio:.6f}")

    sum_small = np.sum(grad_small)
    sum_large = np.sum(grad_large)
    ratio = sum_small / sum_large
    print(f"Gradient ratio (small/large): {ratio:.6f}")
    
    # The ratio should be close to 1 since we're properly scaling by batch size
    ratio_error = abs(ratio - 1.0)
    print(f"Ratio error: {ratio_error:.6f}")
    assert ratio_error < 1e-3, f"Gradient ratio {ratio:.6f} should be close to 1.0"