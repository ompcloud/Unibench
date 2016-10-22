
set(SRC_DIR src)

set(SRC_FILES
  ${SRC_DIR}/correlation.c
)

add_executable(CORR ${SRC_FILES})
target_link_libraries(CORR m)
add_dependencies(Polybench CORR)

add_dependencies(supported CORR)

add_test(Polybench_CORR CORR)
