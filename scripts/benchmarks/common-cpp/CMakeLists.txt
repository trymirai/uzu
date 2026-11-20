cmake_minimum_required(VERSION 3.31)

project(benchmarks_common LANGUAGES C CXX)

include(dependencies.cmake)
benchmark_dependency(glaze)

find_library(IOKIT_FRAMEWORK IOKit REQUIRED)
find_library(COREFOUNDATION_FRAMEWORK CoreFoundation REQUIRED)
find_library(IOREPORT_LIBRARY IOReport REQUIRED)

add_library(${PROJECT_NAME} INTERFACE)

target_sources(${PROJECT_NAME} INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/apple_temp_sensors.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/common.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/energy_counters.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/memory_counters.c"
)

target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_23)

target_include_directories(${PROJECT_NAME} INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/src")

target_link_libraries(
    ${PROJECT_NAME}
    INTERFACE glaze::glaze "${IOKIT_FRAMEWORK}" "${COREFOUNDATION_FRAMEWORK}" "${IOREPORT_LIBRARY}"
)
