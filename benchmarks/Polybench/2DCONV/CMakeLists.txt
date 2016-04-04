
set(SRC_DIR src)

set(SRC_FILES
  ${SRC_DIR}/2DConvolution.c
)

add_executable(2DConvolution ${SRC_FILES})

add_test(Polybench_2DConvolution 2DConvolution)
