
set(SRC_FILES
  src/vector-product_gpu.c
)

add_executable(vector-product ${SRC_FILES})

add_test(vector-product vector-product)
