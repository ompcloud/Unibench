cmake_minimum_required(VERSION 2.6)
project(UniBench)

macro(use_c99)
  if (CMAKE_VERSION VERSION_LESS "3.1")
    if (CMAKE_C_COMPILER_ID STREQUAL "GNU")
      set (CMAKE_C_FLAGS "--std=gnu99 ${CMAKE_C_FLAGS}")
    endif ()
  else ()
    set (CMAKE_C_STANDARD 99)
  endif ()
endmacro()

option(RUN_TEST "Compare with CPU execution" OFF)
if(RUN_TEST)
    message("Run test to compare results with CPU execution")
    add_definitions(-DRUN_TEST=1)
endif()

option(RUN_BENCHMARK "Use large datasets" OFF)
if(RUN_BENCHMARK)
    add_definitions(-DRUN_BENCHMARK=1)
endif()

option(USE_SPARSE "Use sparse datasets" OFF)
if(USE_SPARSE)
    add_definitions(-DSPARSE=1)
endif()

use_c99()

option(OMP_DISABLE "Disable OpenMP" OFF)

if(NOT OMP_DISABLE)
    find_package(OpenMP)
    if(OPENMP_FOUND)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
endif()

# Set OMP target flags
set(OMP_TARGETS "x86_64-unknown-linux-spark")
set(OMP_OFFLOAD_FLAGS "-omptargets=${OMP_TARGETS} --target=x86_64-unknown-linux-gnu")

option(OMP_OFFLOAD "Offload to accelerator" ON)
if(OMP_OFFLOAD AND OPENMP_FOUND)
    # Override C and CXX compiler flags to enable OpenMP offloading
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OMP_OFFLOAD_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OMP_OFFLOAD_FLAGS}")
endif()

# Define the ID of the device
add_definitions(-DDEVICE_ID=0)

# Use ctest for benchmarking
enable_testing()

add_custom_target(supported)

add_subdirectory(benchmarks)
