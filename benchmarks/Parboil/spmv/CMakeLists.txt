
set(SRC_FILES
  src/convert_dataset.c
  src/file.c
  src/main.c
  src/mmio.c
)

add_executable(spmv ${SRC_FILES})
target_link_libraries(spmv parboil m)

file(COPY input DESTINATION input)

add_test(spmv spmv "-i" "input/Dubcova3.mtx.bin,input/vector.bin" "-o" "output.out")
