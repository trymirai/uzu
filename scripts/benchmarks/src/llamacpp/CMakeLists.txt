cmake_minimum_required(VERSION 3.24)

# project setup
project(benchmark_llamacpp LANGUAGES C CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# llama.cpp dependency
include(FetchContent)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries")
set(LLAMA_BUILD_COMMON ON CACHE BOOL "Build common utilities")
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "Build upstream tests")
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "Build upstream examples")
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "Build upstream tools")
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "Build upstream server")
set(LLAMA_BUILD_APP OFF CACHE BOOL "Build upstream application")
FetchContent_Declare(
    llama_cpp
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    GIT_TAG v0.4.1
)
FetchContent_MakeAvailable(llama_cpp)

add_library(mach_memory STATIC ../mach/memory_counters.c)
target_include_directories(mach_memory PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../mach)

add_executable(benchmark_llamacpp main.cpp)
target_link_libraries(benchmark_llamacpp PRIVATE llama llama-common mach_memory)
