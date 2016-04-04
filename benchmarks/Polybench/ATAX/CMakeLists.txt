
set(SRC_FILES
  src/atax.c
)

add_executable(ATAX ${SRC_FILES})

add_test(Polybench_ATAX ATAX)
