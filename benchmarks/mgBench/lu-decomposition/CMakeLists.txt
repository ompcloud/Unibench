
set(SRC_DIR src)

set(SRC_FILES
  ${SRC_DIR}/lu-decomposition_gpu.c
)

add_executable(lu-decomposition ${SRC_FILES})
add_dependencies(mgBench lu-decomposition)

add_dependencies(supported cholesky)

add_test(mgBench_lu-decomposition lu-decomposition)
