
set(SRC_FILES
  src/gemm.c
)

add_executable(GEMM ${SRC_FILES})

add_test(GEMM GEMM)
