
set(SRC_DIR src)

set(SRC_FILES
  ${SRC_DIR}/collinear-list_gpu.c
)

add_executable(collinear-list ${SRC_FILES})

add_test(mgBench_collinear-list collinear-list)
