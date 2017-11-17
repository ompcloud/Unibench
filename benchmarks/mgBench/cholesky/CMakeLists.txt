set(SRC_DIR src)

set(SRC_FILES
  ${SRC_DIR}/cholesky_gpu.c
)

add_executable(cholesky ${SRC_FILES})
target_link_libraries(cholesky m)
add_dependencies(mgBench cholesky)

#add_dependencies(supported cholesky)

add_test(mgBench_cholesky cholesky)
