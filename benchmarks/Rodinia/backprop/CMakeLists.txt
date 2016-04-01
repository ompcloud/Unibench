
set(SRC_FILES
  src/backprop.c
  src/backprop_kernel.c
  src/facetrain.c
  src/imagenet.c
)

add_executable(backprop ${SRC_FILES})
target_link_libraries(backprop m)
