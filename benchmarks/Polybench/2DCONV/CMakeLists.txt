
set(SRC_DIR src)

set(SRC_FILES
  ${SRC_DIR}/2DConvolution.c
)

add_executable(2DCONV ${SRC_FILES})

add_test(Polybench_2DCONV 2DCONV)
