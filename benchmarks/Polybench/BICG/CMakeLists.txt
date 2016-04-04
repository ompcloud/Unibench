
set(SRC_DIR src)

set(SRC_FILES
  ${SRC_DIR}/bicg.c
)

add_executable(BICG ${SRC_FILES})

add_test(Polybench_BICG BICG)
