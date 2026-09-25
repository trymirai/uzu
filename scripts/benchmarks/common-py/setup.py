from setuptools import Extension, setup

setup(
    ext_modules=[
        Extension(
            "mach._mach",
            sources=["src/mach/_mach.c", "../common-cpp/src/memory_counters.c"],
            include_dirs=["../common-cpp/src"],
            depends=["../common-cpp/src/memory_counters.h"],
        )
    ]
)
