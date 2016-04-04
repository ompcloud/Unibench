
set(SRC_FILES
  src/floyd_gpu.c
)

add_executable(floyd ${SRC_FILES})

add_test(mgBench_floyd floyd)
