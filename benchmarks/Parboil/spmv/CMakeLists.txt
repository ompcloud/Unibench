
set(SRC_DIR src)

include_directories(
  ${SRC_DIR}
)

set(SRC_FILES
  ${SRC_DIR}/convert_dataset.c
  ${SRC_DIR}/file.c
  ${SRC_DIR}/main.c
  ${SRC_DIR}/mmio.c
)

add_executable(spmv ${SRC_FILES})
target_link_libraries(spmv parboil m)

file(COPY input DESTINATION .)

add_test(Parboil_spmv spmv "-i" "input/Dubcova3.mtx.bin,input/vector.bin" "-o" "output.out")
