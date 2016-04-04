
set(SRC_DIR src)

set(SRC_FILES
  ${SRC_DIR}/backprop.c
  ${SRC_DIR}/backprop_kernel.c
  ${SRC_DIR}/facetrain.c
  ${SRC_DIR}/imagenet.c
)

add_executable(backprop ${SRC_FILES})
target_link_libraries(backprop m)

add_test(Rodinia_backprop backprop 65536)
