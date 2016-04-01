
set(SRC_FILES
  src/main.cc
  src/io.cc
)

add_executable(sgemm ${SRC_FILES})
target_link_libraries(sgemm parboil)
