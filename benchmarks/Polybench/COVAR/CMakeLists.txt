
set(SRC_FILES
  src/covariance.c
)

add_executable(COVAR ${SRC_FILES})

add_test(Polybench_COVAR COVAR)
