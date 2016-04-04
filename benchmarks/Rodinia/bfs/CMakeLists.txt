
set(SRC_DIR src)

set(SRC_FILES
  ${SRC_DIR}/bfs.c
)

add_executable(bfs ${SRC_FILES})

file(COPY input DESTINATION .)

add_test(Rodinia_bfs bfs 4 "input/graph1MW_6.txt")
