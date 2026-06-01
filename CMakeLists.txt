cmake_minimum_required(VERSION 3.26)
project(bag_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BAG_ENABLE_KNN_FINE_TIMING "Collect per-event kNN timing counters inside the hot query loop" OFF)

add_executable(
    bag_cpp
    src/main.cpp
    src/graph.cpp
    src/loader.cpp
    src/distance.cpp
    src/diagnostics.cpp
    src/factorized_transfer.cpp
    src/object.cpp
    src/partition.cpp
    src/partition_cache.cpp
    src/index.cpp
    src/update_repro.cpp
    src/utils.cpp
)

target_include_directories(bag_cpp PRIVATE src)
target_compile_definitions(
    bag_cpp
    PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
)
if(BAG_ENABLE_KNN_FINE_TIMING)
    target_compile_definitions(bag_cpp PRIVATE BAG_ENABLE_KNN_FINE_TIMING=1)
else()
    target_compile_definitions(bag_cpp PRIVATE BAG_ENABLE_KNN_FINE_TIMING=0)
endif()
