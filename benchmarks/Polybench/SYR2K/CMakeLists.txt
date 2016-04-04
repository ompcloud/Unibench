
set(SRC_DIR src)

set(SRC_FILES
  ${SRC_DIR}/syr2k.c
)

add_executable(SYR2K ${SRC_FILES})

add_test(Polybench_SYR2K SYR2K)
