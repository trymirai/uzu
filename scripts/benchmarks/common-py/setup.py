from setuptools import Extension, setup

setup(
    ext_modules=[
        Extension(
            "mach._native",
            sources=[
                "../common-cpp/src/memory_counters.c",
                "../common-cpp/src/apple_temp_sensors.c",
                "../common-cpp/src/energy_counters.c",
            ],
            include_dirs=["../common-cpp/src"],
            depends=[
                "../common-cpp/src/memory_counters.h",
                "../common-cpp/src/apple_temp_sensors.h",
                "../common-cpp/src/energy_counters.h",
            ],
            extra_link_args=["-framework", "IOKit", "-framework", "CoreFoundation", "-lIOReport"],
        )
    ],
)
