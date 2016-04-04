
set(SRC_FILES
  src/gramschmidt.c
)

add_executable(GRAMSCHM ${SRC_FILES})
target_link_libraries(GRAMSCHM m)

add_test(Polybench_GRAMSCHM GRAMSCHM)
