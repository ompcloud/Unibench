
set(SRC_FILES
  src/mat-mul_gpu.c
)

add_executable(mat-mul ${SRC_FILES})

add_test(mgBench_mat-mul mat-mul)
