
set(SRC_FILES
  src/correlation.c
)

add_executable(CORR ${SRC_FILES})
target_link_libraries(CORR m)

add_test(Polybench_CORR CORR)
