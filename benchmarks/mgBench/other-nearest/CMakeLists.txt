
set(SRC_DIR src)

set(SRC_FILES
  ${SRC_DIR}/other-nearest_gpu.c
)

add_executable(other-nearest ${SRC_FILES})
target_link_libraries(other-nearest m)

add_test(mgBench_other-nearest other-nearest)
