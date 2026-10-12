"""Collect the engine-specific classes with pytest's default file discovery."""

from .engines.llamacpp import TestLlamaCpp
from .engines.mldrift import TestMLDrift
from .engines.mlx import TestMLX
from .engines.mlxserve import TestMLXServe
from .engines.mtplx import TestMTPLX
from .engines.omlx import TestOMLX
from .engines.splash import TestSplash
from .engines.uzu import TestUzu

__all__ = ["TestLlamaCpp", "TestMLDrift", "TestMLX", "TestMLXServe", "TestMTPLX", "TestOMLX", "TestSplash", "TestUzu"]
