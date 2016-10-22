
set(SRC_DIR src)

set(SRC_FILES
  ${SRC_DIR}/atax.c
)

add_executable(ATAX ${SRC_FILES})
add_dependencies(Polybench ATAX)

add_dependencies(supported 3DCONV)

add_test(Polybench_ATAX ATAX)
