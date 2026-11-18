import sys

from setuptools import Extension, setup

# The memory counters read mach task info and IOKit sensors, which exist on macOS only.
setup(
    ext_modules=[
        Extension(
            "mach._mach",
            sources=[
                "src/mach/_mach.c",
                "../common-cpp/src/memory_counters.c",
                "../common-cpp/src/apple_temp_sensors.c",
            ],
            include_dirs=["../common-cpp/src"],
            depends=["../common-cpp/src/memory_counters.h", "../common-cpp/src/apple_temp_sensors.h"],
            extra_link_args=["-framework", "IOKit"],
        )
    ]
    if sys.platform == "darwin"
    else []
)
