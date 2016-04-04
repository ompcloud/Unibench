
set(SRC_FILES
  src/search-vector_gpu.c
)

add_executable(search-vector ${SRC_FILES})

add_test(mgBench_search-vector search-vector)
