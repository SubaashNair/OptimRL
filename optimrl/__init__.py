from .core import GRPO

__version__ = "0.1.0"
__all__ = ["GRPO"]
from . import _version
from ._version import get_versions
__version__ = _version.get_versions()['version']
del get_versions
