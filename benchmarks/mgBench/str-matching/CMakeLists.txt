
set(SRC_DIR src)

set(SRC_FILES
  ${SRC_DIR}/str-matching_gpu.c
)

add_executable(str-matching ${SRC_FILES})
add_dependencies(mgBench str-matching)

#add_dependencies(supported str-matching)

#add_test(mgBench_str-matching str-matching)
