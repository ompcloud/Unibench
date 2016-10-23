cmake_minimum_required(VERSION 2.6)
project(UniBench)

option(RUN_TEST "Compare with CPU execution" OFF)

if(RUN_TEST)
    message("Run test to compare results with CPU execution")
    add_definitions(-DRUN_TEST=1)
endif(RUN_TEST)

# Set OMP target flags
set(OMP_TARGETS "x86_64-unknown-linux-spark")
set(OMP_OFFLOAD_FLAGS "-fopenmp -omptargets=${OMP_TARGETS} --target=x86_64-unknown-linux-gnu")

# Override C and CXX compiler flags to enable OpenMP offloading
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OMP_OFFLOAD_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OMP_OFFLOAD_FLAGS}")

# Define the ID of the device
add_definitions(-DDEVICE_ID=0)

# Use ctest for benchmarking
enable_testing()

add_subdirectory(benchmarks)

#add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} --verbose)
