
set(SRC_DIR src)

set(SRC_FILES
  ${SRC_DIR}/floyd_gpu.c
)

add_executable(floyd ${SRC_FILES})
add_dependencies(mgBench floyd)

add_dependencies(supported floyd)

add_test(mgBench_floyd floyd)
