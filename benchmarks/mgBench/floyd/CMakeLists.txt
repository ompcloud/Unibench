
set(SRC_DIR src)

set(SRC_FILES
  ${SRC_DIR}/floyd_gpu.c
)

add_executable(floyd ${SRC_FILES})

add_test(mgBench_floyd floyd)
