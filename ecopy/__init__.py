from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("ecopy2")
except PackageNotFoundError:
    __version__ = "unknown"

from .regression import *
from .ordination import *
from .diversity import *
from .matrix_comp import *
from .base_funcs import *
from .utils import load_data
