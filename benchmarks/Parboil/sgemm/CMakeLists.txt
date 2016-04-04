
set(SRC_FILES
  src/main.cc
  src/io.cc
)

add_executable(sgemm ${SRC_FILES})
target_link_libraries(sgemm parboil)

file(COPY input DESTINATION input)

add_test(sgemm sgemm "-i" "input/matrix1.txt,input/matrix2.txt,input/matrix2t.txt" "-o" "matrix3.txt")
