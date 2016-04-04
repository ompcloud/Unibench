
set(SRC_DIR src)

include_directories(
  ${SRC_DIR}
)

set(SRC_FILES
  ${SRC_DIR}/file.c
  ${SRC_DIR}/kernels.c
  ${SRC_DIR}/main.c
)

add_executable(stencil ${SRC_FILES})
target_link_libraries(stencil parboil)

file(COPY input DESTINATION input)

add_test(Parboil_stencil stencil "-i" "input/512x512x64x100.bin" "-o" "512x512x64.out" "--" "512" "512" "64" "100")
