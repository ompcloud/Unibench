
set(SRC_FILES
  src/2mm.c
)

add_executable(2MM ${SRC_FILES})

add_test(Polybench_2MM 2MM)
