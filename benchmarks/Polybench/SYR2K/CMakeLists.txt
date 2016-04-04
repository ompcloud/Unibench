
set(SRC_FILES
  src/syr2k.c
)

add_executable(SYR2K ${SRC_FILES})

add_test(Polybench_SYR2K SYR2K)
