
set(SRC_FILES
  src/lu-decomposition_gpu.c
)

add_executable(lu-decomposition ${SRC_FILES})

#add_test(mgBench_lu-decomposition lu-decomposition)
