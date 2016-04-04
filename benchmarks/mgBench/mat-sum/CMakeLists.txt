
set(SRC_FILES
  src/mat-sum_gpu.c
)

add_executable(mat-sum ${SRC_FILES})

add_test(mgBench_mat-sum mat-sum)
