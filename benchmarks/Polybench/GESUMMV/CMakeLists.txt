
set(SRC_FILES
  src/gesummv.c
)

add_executable(GESUMMV ${SRC_FILES})

add_test(Polybench_GESUMMV GESUMMV)
