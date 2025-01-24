import ctypes
import numpy as np
import os
import platform
import sys
from numpy.ctypeslib import ndpointer

class GRPO:
    """
    Group Relative Policy Optimization (GRPO) implementation.
    
    This class provides a Python interface to the C implementation of GRPO,
    handling all necessary memory management and type conversions.
    """
    
    def __init__(self, epsilon=0.2, beta=0.1):
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
        
        # Load the C library
        self._load_library()
        
        # Set up C function interfaces
        self._setup_functions()
    
    def _find_library_path(self):
        """Find the path to the compiled library file."""
        # Get the directory of the current file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Determine library name based on platform
        if platform.system() == 'Darwin':
            lib_name = 'libgrpo.dylib'
        elif platform.system() == 'Linux':
            lib_name = 'libgrpo.so'
        elif platform.system() == 'Windows':
            lib_name = 'libgrpo.dll'
        else:
            raise OSError(f"Unsupported operating system: {platform.system()}")
        
        # Possible locations for the library
        possible_paths = [
            os.path.join(current_dir, 'c_src', lib_name),
            os.path.join(current_dir, lib_name),
            os.path.join(os.path.dirname(current_dir), 'c_src', lib_name),
            os.path.join(sys.prefix, 'lib', lib_name)
        ]
        
        # Find the first existing library file
        for path in possible_paths:
            if os.path.exists(path):
                return path
                
        paths_str = '\n'.join(possible_paths)
        raise FileNotFoundError(
            f"Could not find {lib_name}. Searched in:\n{paths_str}\n"
            "Try reinstalling the package or building the C extension."
        )
    
    def _load_library(self):
        """Load the compiled C library."""
        try:
            lib_path = self._find_library_path()
            self.lib = ctypes.CDLL(lib_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load GRPO C library: {str(e)}") from e
    
    def _setup_functions(self):
        """Configure C function interfaces with proper type annotations."""
        # Define the batch structure for C
        class GRPOBatch(ctypes.Structure):
            _fields_ = [
                ('log_probs_old', ctypes.POINTER(ctypes.c_double)),
                ('log_probs_ref', ctypes.POINTER(ctypes.c_double)),
                ('rewards', ctypes.POINTER(ctypes.c_double)),
                ('group_size', ctypes.c_int)
            ]
        
        self.GRPOBatch = GRPOBatch
        
        # Set up compute_advantages function interface
        self.lib.compute_advantages.argtypes = [
            ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),
            ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),
            ctypes.c_int
        ]
        self.lib.compute_advantages.restype = None
        
        # Set up grpo_loss function interface
        self.lib.grpo_loss.argtypes = [
            ctypes.POINTER(GRPOBatch),
            ndpointer(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ndpointer(ctypes.c_double),
            ctypes.c_double,
            ctypes.c_double
        ]
        self.lib.grpo_loss.restype = None
    
    def compute_loss(self, batch_data, log_probs_new):
        """
        Compute GRPO loss and gradients.
        
        Args:
            batch_data (dict): Dictionary containing:
                - log_probs_old: Array of old policy log probabilities
                - log_probs_ref: Array of reference policy log probabilities
                - rewards: Array of rewards
                - group_size: Integer size of the group
            log_probs_new: Array of new policy log probabilities
            
        Returns:
            tuple: (loss value, gradients array)
        """
        # Input validation
        required_keys = {'log_probs_old', 'log_probs_ref', 'rewards', 'group_size'}
        missing_keys = required_keys - set(batch_data.keys())
        if missing_keys:
            raise KeyError(f"Missing required keys: {missing_keys}")
            
        for key in ['log_probs_old', 'log_probs_ref', 'rewards']:
            if not isinstance(batch_data[key], np.ndarray):
                batch_data[key] = np.array(batch_data[key], dtype=np.float64)
            elif batch_data[key].dtype != np.float64:
                batch_data[key] = batch_data[key].astype(np.float64)
        
        if not isinstance(log_probs_new, np.ndarray):
            log_probs_new = np.array(log_probs_new, dtype=np.float64)
        elif log_probs_new.dtype != np.float64:
            log_probs_new = log_probs_new.astype(np.float64)
        
        # Validate shapes
        group_size = batch_data['group_size']
        if len(log_probs_new) != group_size:
            raise ValueError(f"log_probs_new length ({len(log_probs_new)}) must match group_size ({group_size})")
        
        # Prepare batch structure for C
        batch = self.GRPOBatch(
            batch_data['log_probs_old'].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            batch_data['log_probs_ref'].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            batch_data['rewards'].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(group_size)
        )
        
        # Prepare output arrays
        loss = ctypes.c_double(0.0)
        grad = np.zeros_like(log_probs_new, dtype=np.float64)
        
        # Call C function
        self.lib.grpo_loss(
            ctypes.byref(batch),
            log_probs_new,
            ctypes.byref(loss),
            grad,
            ctypes.c_double(self.epsilon),
            ctypes.c_double(self.beta)
        )
        
        return loss.value, grad