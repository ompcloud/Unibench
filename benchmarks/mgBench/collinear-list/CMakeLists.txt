
set(SRC_FILES
  src/collinear-list_gpu.c
)

add_executable(collinear-list ${SRC_FILES})

add_test(collinear-list collinear-list)
