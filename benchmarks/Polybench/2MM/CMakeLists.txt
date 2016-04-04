
set(SRC_DIR src)

set(SRC_FILES
  ${SRC_DIR}/2mm.c
)

add_executable(2MM ${SRC_FILES})
add_dependencies(Polybench 2MM)

add_test(Polybench_2MM 2MM)
