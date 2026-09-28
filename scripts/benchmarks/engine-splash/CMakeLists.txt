cmake_minimum_required(VERSION 3.31)

# Match the minimum OS and compiler settings used by Splash's production build.
set(CMAKE_OSX_DEPLOYMENT_TARGET "26.4" CACHE STRING "Minimum supported macOS version")
project(engine-splash LANGUAGES C CXX OBJCXX)

# CMake's Release defaults add NDEBUG; upstream uses just the explicit flags
# below, so retain the same assertions and optimization in the binding.
set(CMAKE_C_FLAGS_RELEASE "")
set(CMAKE_CXX_FLAGS_RELEASE "")
set(CMAKE_OBJCXX_FLAGS_RELEASE "")

if(NOT APPLE OR NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    message(FATAL_ERROR "Splash requires an Apple Silicon Mac")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(SPLASH_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps/splash")
set(SPLASH_BUILD_DIR "${SPLASH_SOURCE_DIR}/build")
set(SPLASH_LIB_DIR "${CMAKE_CURRENT_BINARY_DIR}/lib")
set(SPLASH_BUILD_JOBS "8" CACHE STRING "Parallel jobs for the upstream Splash build")

if(NOT EXISTS "${SPLASH_SOURCE_DIR}/runtime/engine/Bootstrap.hpp")
    message(FATAL_ERROR "Splash sources are missing; run engine-splash/bootstrap.sh first")
endif()

find_program(SPLASH_MAKE make REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(IOKIT_FRAMEWORK IOKit REQUIRED)

# Keep all production sources and build rules upstream. Always run make so its
# own dependency, compiler configuration and build identity checks remain active.
add_custom_target(splash-upstream
    COMMAND "${SPLASH_MAKE}" -C "${SPLASH_SOURCE_DIR}" --no-print-directory
        -j "${SPLASH_BUILD_JOBS}" "BUILD_ID_PYTHON=${Python3_EXECUTABLE}"
        build/engine/libsplash.a build/engine/BuildIdentity.hpp build/splash.metallib
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${SPLASH_LIB_DIR}"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "${SPLASH_BUILD_DIR}/splash.metallib" "${SPLASH_LIB_DIR}/splash.metallib"
    BYPRODUCTS
        "${SPLASH_BUILD_DIR}/engine/libsplash.a"
        "${SPLASH_BUILD_DIR}/engine/BuildIdentity.hpp"
        "${SPLASH_BUILD_DIR}/splash.metallib"
    USES_TERMINAL
    VERBATIM
)

# Only production engine sources are linked; the executable's host, fd
# transport and main are not part of the in-process extension.
Python3_add_library(_splash_native MODULE WITH_SOABI
    src/bindings.cpp
    src/engine.mm
)
add_dependencies(_splash_native splash-upstream)
set_target_properties(_splash_native PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${SPLASH_LIB_DIR}"
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    OBJCXX_STANDARD 20
    OBJCXX_STANDARD_REQUIRED ON
    OBJCXX_EXTENSIONS OFF
)
target_compile_options(_splash_native PRIVATE
    -O3 -Wall -Wextra -Werror
    "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>"
)
set_source_files_properties(src/engine.mm PROPERTIES
    COMPILE_OPTIONS "-include;${SPLASH_BUILD_DIR}/engine/BuildIdentity.hpp"
)
target_include_directories(_splash_native PRIVATE
    src
    "${SPLASH_SOURCE_DIR}/runtime"
    "${SPLASH_BUILD_DIR}/engine"
)
target_link_libraries(_splash_native PRIVATE
    "${SPLASH_BUILD_DIR}/engine/libsplash.a"
    "${FOUNDATION_FRAMEWORK}"
    "${METAL_FRAMEWORK}"
    "${IOKIT_FRAMEWORK}"
)
