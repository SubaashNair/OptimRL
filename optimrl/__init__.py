from .core import GRPO
from .agents import create_agent, BaseAgent, GRPOAgent, ContinuousGRPOAgent
from ._version import get_versions

__all__ = ['GRPO', 'create_agent', 'BaseAgent', 'GRPOAgent', 'ContinuousGRPOAgent']

__version__ = get_versions()["version"]
del get_versions
