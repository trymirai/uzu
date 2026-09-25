cmake_minimum_required(VERSION 3.31)

project(benchmarks_common LANGUAGES C CXX)

include(FetchContent)

FetchContent_Declare(
    glaze
    GIT_REPOSITORY https://github.com/stephenberry/glaze.git
    GIT_TAG v9.0.0
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(glaze)

add_library(${PROJECT_NAME} INTERFACE)

target_sources(${PROJECT_NAME} INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/memory_counters.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/common.cpp"
)

target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_23)

target_include_directories(${PROJECT_NAME} INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/src")

target_link_libraries(${PROJECT_NAME} INTERFACE glaze::glaze)
