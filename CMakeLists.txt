cmake_minimum_required(VERSION 2.6)
project(UniBench)

set(OMP_TARGETS "x86_64-unknown-linux-spark")
set(OMP_OFFLOAD_FLAGS "-fopenmp -omptargets=${OMP_TARGETS}")

# Override C and CXX compiler flags to enable OpenMP offloading
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OMP_OFFLOAD_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OMP_OFFLOAD_FLAGS}")

enable_testing()

add_subdirectory(benchmarks)

#add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} --verbose)

configure_file(unibench2 unibench2 COPYONLY)
