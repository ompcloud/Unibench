
set(SRC_FILES
  src/cholesky_gpu.c
)

add_executable(cholesky ${SRC_FILES})
target_link_libraries(cholesky m)

add_test(cholesky cholesky)
