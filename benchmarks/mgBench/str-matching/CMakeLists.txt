
set(SRC_FILES
  src/str-matching_gpu.c
)

add_executable(str-matching ${SRC_FILES})

add_test(mgBench_str-matching str-matching)
