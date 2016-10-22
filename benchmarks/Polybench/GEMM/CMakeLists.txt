
set(SRC_DIR src)

set(SRC_FILES
  ${SRC_DIR}/gemm.c
)

add_executable(GEMM ${SRC_FILES})
add_dependencies(Polybench GEMM)

add_dependencies(experiments GEMM)
add_dependencies(supported GEMM)

add_test(Polybench_GEMM GEMM)
