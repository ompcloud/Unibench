
set(SRC_DIR src)

set(SRC_FILES
  ${SRC_DIR}/k-nearest_gpu.c
)

add_executable(k-nearest ${SRC_FILES})
target_link_libraries(k-nearest m)

add_test(mgBench_k-nearest k-nearest)
