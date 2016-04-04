
set(SRC_FILES
  src/k-nearest_gpu.c
)

add_executable(k-nearest ${SRC_FILES})
target_link_libraries(k-nearest m)

add_test(k-nearest k-nearest)
