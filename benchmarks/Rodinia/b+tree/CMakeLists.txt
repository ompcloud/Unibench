
set(SRC_FILES
  src/main.c
  src/util/num/num.c
  src/util/timer/timer.c
  src/kernel/kernel_cpu.c
  src/kernel/kernel_cpu_2.c
)

add_executable(btree ${SRC_FILES})
target_link_libraries(btree m)
