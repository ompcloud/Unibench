
set(SRC_FILES
  src/bfs.c
)

add_executable(bfs ${SRC_FILES})

file(COPY input DESTINATION input)

add_test(Rodinia_bfs bfs 4 "input/graph1MW_6.txt")
