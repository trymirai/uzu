cmake_minimum_required(VERSION 3.31)
include(FetchContent)

# project setup
project(llamacpp_bench LANGUAGES C CXX)
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
FetchContent_Declare(
    llama_cpp
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    GIT_TAG v0.4.1
)
FetchContent_MakeAvailable(llama_cpp)

# CLI11
FetchContent_Declare(
    cli11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG v2.7.2
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(cli11)

# Glaze
FetchContent_Declare(
    glaze
    GIT_REPOSITORY https://github.com/stephenberry/glaze.git
    GIT_TAG v8.4.0
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(glaze)

# memory counters
add_library(mach_memory STATIC ../mach/memory_counters.c)
target_include_directories(mach_memory PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../mach)

# sources
add_executable(${PROJECT_NAME} 
    main.cpp 
    common.cpp
    runner.cpp
)

# libraries
target_link_libraries(${PROJECT_NAME}  
    PRIVATE 
        llama 
        llama-common 
        CLI11::CLI11
        glaze::glaze
        mach_memory
)
