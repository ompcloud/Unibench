
set(SRC_DIR src)

set(SRC_FILES
  ${SRC_DIR}/mvt.c
)

add_executable(MVT ${SRC_FILES})

add_test(Polybench_MVT MVT)
