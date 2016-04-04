
set(SRC_FILES
  src/file.c
  src/kernels.c
  src/main.c
)

add_executable(stencil ${SRC_FILES})
target_link_libraries(stencil parboil)

file(COPY input DESTINATION input)

add_test(stencil stencil "-i" "input/512x512x64x100.bin" "-o" "512x512x64.out" "--" "512" "512" "64" "100")
