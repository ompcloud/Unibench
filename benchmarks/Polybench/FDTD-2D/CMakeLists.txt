
set(SRC_DIR src)

set(SRC_FILES
  ${SRC_DIR}/fdtd2d.c
)

add_executable(FDTD-2D ${SRC_FILES})

add_test(Polybench_FDTD-2D FDTD-2D)
