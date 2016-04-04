
set(SRC_DIR src)

set(SRC_FILES
  ${SRC_DIR}/3DConvolution.c
)

add_executable(3DCONV ${SRC_FILES})
add_dependencies(Polybench 3DCONV)

add_test(Polybench_3DCONV 3DCONV)
