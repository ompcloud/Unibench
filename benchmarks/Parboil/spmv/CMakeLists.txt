
set(SRC_FILES
  src/convert_dataset.c
  src/file.c
  src/main.c
  src/mmio.c
)

add_executable(spmv ${SRC_FILES})
target_link_libraries(spmv parboil m)
