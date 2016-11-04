
set(SRC_DIR src)

set(SRC_FILES
  ${SRC_DIR}/main.cc
  ${SRC_DIR}/io.cc
)

add_executable(sgemm ${SRC_FILES})
target_link_libraries(sgemm parboilutil)
add_dependencies(Parboil sgemm)

file(COPY input DESTINATION .)

add_test(Parboil_sgemm sgemm "-i" "input/matrix1.txt,input/matrix2.txt,input/matrix2t.txt" "-o" "matrix3.txt")
