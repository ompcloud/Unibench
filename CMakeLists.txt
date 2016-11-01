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
endmacro(use_c99)

option(RUN_TEST "Compare with CPU execution" OFF)
if(RUN_TEST)
    message("Run test to compare results with CPU execution")
    add_definitions(-DRUN_TEST=1)
endif(RUN_TEST)

option(RUN_BENCHMARK "Use large datasets" OFF)
if(RUN_BENCHMARK)
    add_definitions(-DRUN_BENCHMARK=1)
endif(RUN_BENCHMARK)

use_c99()

find_package(OpenMP)

# Set OMP target flags
set(OMP_TARGETS "x86_64-unknown-linux-spark")
set(OMP_OFFLOAD_FLAGS "-omptargets=${OMP_TARGETS} --target=x86_64-unknown-linux-gnu")

option(OMP_OFFLOAD "Offload to accelerator" ON)
if(OMP_OFFLOAD)
    # Override C and CXX compiler flags to enable OpenMP offloading
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OMP_OFFLOAD_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OMP_OFFLOAD_FLAGS}")
endif(OMP_OFFLOAD)

# Define the ID of the device
add_definitions(-DDEVICE_ID=0)

# Use ctest for benchmarking
enable_testing()

add_subdirectory(benchmarks)

#add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} --verbose)
