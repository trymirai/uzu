from setuptools import Extension, setup

setup(
    ext_modules=[
        Extension(
            "mach._mach",
            sources=["src/mach/_mach.c", "src/mach/memory_counters.c"],
            depends=["src/mach/memory_counters.h"],
        )
    ]
)
