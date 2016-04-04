
set(SRC_DIR src)

set(SRC_FILES
  ${SRC_DIR}/3mm.c
)

add_executable(3MM ${SRC_FILES})

add_test(Polybench_3MM 3MM)
