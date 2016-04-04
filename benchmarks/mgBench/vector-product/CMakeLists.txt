
set(SRC_FILES
  src/vector-product_gpu.c
)

add_executable(vector-product ${SRC_FILES})

add_test(mgBench_vector-product vector-product)
