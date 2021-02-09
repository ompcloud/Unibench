# cmake -D CMAKE_C_COMPILER=clang -D CMAKE_CXX_COMPILER=clang++
cmake_minimum_required(VERSION 3.10)
project(UniBench)

# macro(use_c99)
#   if (CMAKE_VERSION VERSION_LESS "3.1")
#     if (CMAKE_C_COMPILER_ID STREQUAL "GNU")
#       set (CMAKE_C_FLAGS "--std=gnu99 ${CMAKE_C_FLAGS}")
#     endif ()
#   else ()
#     set (CMAKE_C_STANDARD 99)
#   endif ()
# endmacro()

# options
option(RUN_TEST "Compare with CPU execution" OFF)
option(RUN_BENCHMARK "Use large datasets" OFF)
option(OMP_OFFLOAD "Offload to accelerator" ON)
option(OMP_DISABLE "Disable OpenMP" OFF)

#set (CMAKE_BINARY_DIR ${CMAKE_CURRENT_SOURCE_DIR})
#set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
#message("DIRS: \n${CMAKE_CURRENT_SOURCE_DIR}\n${CMAKE_BINARY_DIR}\n${CMAKE_RUNTIME_OUTPUT_DIRECTORY}.")

if(RUN_TEST)
  if (RUN_BENCHMARK)
    message(FATAL_ERROR "RUN_TEST and RUN_BENCHMARK are mutually exclusive.")
  endif()
endif()

if(RUN_TEST)
  message("Run test to compare results with CPU execution")
  add_definitions(-DRUN_TEST=1)
endif()

if(RUN_BENCHMARK)
  add_definitions(-DRUN_BENCHMARK=1)
endif()

if(NOT OMP_DISABLE)
  find_package(OpenMP)
  if(OPENMP_FOUND)
    message("Found OpenMP.")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  endif()
endif()

# check the architecture, cuda availability, and the compiler
if (CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64" OR  CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
  find_package(CUDA 10.2)
  if(CUDA_FOUND)
    message("Found CUDA.")
    if(OMP_OFFLOAD AND OPENMP_FOUND)
      # Override C and CXX compiler flags to enable OpenMP offloading
      if (CMAKE_C_COMPILER_ID STREQUAL "GNU" OR CMAKE_C_COMPILER_ID STREQUAL "Clang")
        if (CMAKE_C_COMPILER_ID STREQUAL "GNU")
          message("Using GNU compiler.")
          #set(OMP_OFFLOAD_FLAGS "-fopenmp --target=x86_64-unknown-linux-gnu")
          set(OMP_OFFLOAD_FLAGS "-fopenmp")
        else()
          message("Using Clang compiler.")
          set(OMP_OFFLOAD_FLAGS "-fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda")
        endif()
      else()
        message(FATAL_ERROR "Unsupported compiler ${CMAKE_C_COMPILER_ID}. Only GNU and CLang are supported.")
      endif()
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OMP_OFFLOAD_FLAGS}")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OMP_OFFLOAD_FLAGS}")
    endif()
  endif()
else()
  message(FATAL_ERROR "Unsupported architecture ${CMAKE_SYSTEM_PROCESSOR}. Only aarch64 and x86_64 are supported.")
endif()
# Set OMP target flags



# Define the ID of the device
add_definitions(-DDEVICE_ID=0)

# Use ctest for benchmarking
enable_testing()

add_custom_target(supported)

add_subdirectory(benchmarks)
