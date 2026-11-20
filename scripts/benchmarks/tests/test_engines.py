"""Collect the engine-specific classes with pytest's default file discovery."""

from .engines.ios import TestIOS
from .engines.kapitsa import TestKapitsa
from .engines.llamacpp import TestLlamaCpp
from .engines.mlx import TestMLX
from .engines.mlxserve import TestMLXServe
from .engines.mtplx import TestMTPLX
from .engines.omlx import TestOMLX
from .engines.splash import TestSplash
from .engines.uzu import TestUzu

__all__ = [
    "TestIOS",
    "TestKapitsa",
    "TestLlamaCpp",
    "TestMLX",
    "TestMLXServe",
    "TestMTPLX",
    "TestOMLX",
    "TestSplash",
    "TestUzu",
]
