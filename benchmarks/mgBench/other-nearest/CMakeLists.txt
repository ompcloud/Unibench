
set(SRC_FILES
  src/other-nearest_gpu.c
)

add_executable(other-nearest ${SRC_FILES})
target_link_libraries(other-nearest m)
