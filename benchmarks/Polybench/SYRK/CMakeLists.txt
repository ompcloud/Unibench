
set(SRC_DIR src)

set(SRC_FILES
  ${SRC_DIR}/syrk.c
)

add_executable(SYRK ${SRC_FILES})

add_test(Polybench_SYRK SYRK)
