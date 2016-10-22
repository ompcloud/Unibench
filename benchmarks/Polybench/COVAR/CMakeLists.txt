
set(SRC_DIR src)

set(SRC_FILES
  ${SRC_DIR}/covariance.c
)

add_executable(COVAR ${SRC_FILES})
add_dependencies(Polybench COVAR)

add_dependencies(experiments COVAR)
add_dependencies(supported COVAR)

add_test(Polybench_COVAR COVAR)
