
set(SRC_FILES
  src/file.c
  src/kernels.c
  src/main.c
)

add_executable(stencil ${SRC_FILES})
target_link_libraries(stencil parboil)
