
set(SRC_DIR src)

set(SRC_FILES
  ${SRC_DIR}/gesummv.c
)

add_executable(GESUMMV ${SRC_FILES})
add_dependencies(Polybench GESUMMV)

add_test(Polybench_GESUMMV GESUMMV)
