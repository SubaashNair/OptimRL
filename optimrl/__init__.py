from .core import GRPO
from ._version import get_versions

# __version__ = "0.1.0"
__version__ = get_versions()["version"]
del get_versions
