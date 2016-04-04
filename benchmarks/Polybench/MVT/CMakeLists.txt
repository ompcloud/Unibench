
set(SRC_FILES
  src/mvt.c
)

add_executable(MVT ${SRC_FILES})

add_test(Polybench_MVT MVT)
