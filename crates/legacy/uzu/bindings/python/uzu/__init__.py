import sys as _sys

from . import uzu
from ._tool import UzuToolFunction as UzuToolFunction
from ._tool import uzu_tool_function as uzu_tool_function
from .uzu import *  # noqa: F403

_sys.modules.pop("uzu.uzu", None)
del uzu
