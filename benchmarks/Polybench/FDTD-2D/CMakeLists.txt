
set(SRC_FILES
  src/fdtd2d.c
)

add_executable(FDTD-2D ${SRC_FILES})

add_test(Polybench_FDTD-2D FDTD-2D)
