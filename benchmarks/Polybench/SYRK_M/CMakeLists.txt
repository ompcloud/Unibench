
set(SRC_DIR src)

set(SRC_FILES
  ${SRC_DIR}/syrk_m.c
)

add_executable(SYRK_M ${SRC_FILES})

add_test(Polybench_SYRK_M SYRK_M)
