
set(SRC_FILES
  src/bicg.c
)

add_executable(BICG ${SRC_FILES})

add_test(Polybench_BICG BICG)
