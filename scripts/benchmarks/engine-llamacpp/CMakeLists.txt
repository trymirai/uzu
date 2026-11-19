cmake_minimum_required(VERSION 3.31)
include(../common-cpp/dependencies.cmake)

# project setup
project(engine-llamacpp LANGUAGES C CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# llama.cpp
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries")
set(LLAMA_BUILD_COMMON ON CACHE BOOL "Build common utilities")
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "Build upstream tests")
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "Build upstream examples")
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "Build upstream tools")
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "Build upstream server")
set(LLAMA_BUILD_APP OFF CACHE BOOL "Build upstream application")
benchmark_dependency(llama_cpp)

# CLI11
benchmark_dependency(cli11)

# common sources
add_subdirectory(../common-cpp common-cpp)

# sources
set(SOURCES
    src/engine.cpp
    src/main.cpp
    src/util.cpp
)
if(IOS)
    add_library(${PROJECT_NAME} SHARED ${SOURCES})
    set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME BenchmarksLlamacpp)
    target_link_options(${PROJECT_NAME} PRIVATE -Wl,-exported_symbol,_benchmarks_llamacpp_main)
else()
    add_executable(${PROJECT_NAME} ${SOURCES})
endif()

# libraries
target_link_libraries(${PROJECT_NAME}
    PRIVATE
        llama
        llama-common
        CLI11::CLI11
        benchmarks_common
)
