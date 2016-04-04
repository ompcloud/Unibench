
set(SRC_DIR src)

set(SRC_FILES
  ${SRC_DIR}/file.cc
  ${SRC_DIR}/main.c
)

include_directories(
  ${SRC_DIR}
)

add_executable(mri-q ${SRC_FILES})
target_link_libraries(mri-q parboil m)

file(COPY input DESTINATION input)

add_test(Parboil_mri-q mri-q "-i" "input/64_64_64_dataset.bin" "-o" "64_64_64_dataset.out")
