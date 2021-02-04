
set(SRC_DIR src)

set(SRC_FILES
  ${SRC_DIR}/2mm.c
)

message("DIRS2: \n${CMAKE_CURRENT_SOURCE_DIR}\n${CMAKE_BINARY_DIR}\n${CMAKE_RUNTIME_OUTPUT_DIRECTORY}.")

add_executable(2MM ${SRC_FILES})
add_dependencies(Polybench 2MM)

add_dependencies(experiments 2MM)
add_dependencies(supported 2MM)

add_test(Polybench_2MM 2MM)
