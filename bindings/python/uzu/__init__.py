import sys as _sys

from . import uzu
from .uzu import *  # noqa: F403

_sys.modules.pop("uzu.uzu", None)
del uzu
