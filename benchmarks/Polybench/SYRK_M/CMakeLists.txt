
set(SRC_FILES
  src/syrk_m.c
)

add_executable(SYRK_M ${SRC_FILES})

add_test(Polybench_SYRK_M SYRK_M)
