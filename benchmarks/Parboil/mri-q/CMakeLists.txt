set(SRC_DIR src)

set(SRC_FILES
  ${SRC_DIR}/file.cc
  ${SRC_DIR}/main.c
)

add_executable(mri-q ${SRC_FILES})
target_link_libraries(mri-q parboil m)
