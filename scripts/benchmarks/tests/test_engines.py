"""Collect the engine-specific classes with pytest's default file discovery."""

from .engines.llamacpp import TestLlamaCpp
from .engines.mlx import TestMLX
from .engines.mtplx import TestMTPLX
from .engines.omlx import TestOMLX
from .engines.splash import TestSplash
from .engines.uzu import TestUzu

__all__ = ["TestLlamaCpp", "TestMLX", "TestMTPLX", "TestOMLX", "TestSplash", "TestUzu"]
