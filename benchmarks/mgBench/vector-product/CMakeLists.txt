
set(SRC_DIR src)

set(SRC_FILES
  ${SRC_DIR}/vector-product_gpu.c
)

add_executable(vector-product ${SRC_FILES})
add_dependencies(mgBench vector-product)

add_dependencies(supported vector-product)

add_test(mgBench_vector-product vector-product)
