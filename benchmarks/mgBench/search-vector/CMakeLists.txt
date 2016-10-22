
set(SRC_DIR src)

set(SRC_FILES
  ${SRC_DIR}/search-vector_gpu.c
)

add_executable(search-vector ${SRC_FILES})
add_dependencies(mgBench search-vector)

add_dependencies(supported search-vector)

add_test(mgBench_search-vector search-vector)
