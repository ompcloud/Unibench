
set(SRC_FILES
  src/3DConvolution.c
)

add_executable(3DCONV ${SRC_FILES})

add_test(Polybench_3DCONV 3DCONV)
