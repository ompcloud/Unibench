
set(SRC_FILES
  src/main.c
  src/util/num/num.c
  src/util/timer/timer.c
  src/kernel/kernel_cpu.c
  src/kernel/kernel_cpu_2.c
)

add_executable(btree ${SRC_FILES})
target_link_libraries(btree m)

file(COPY input DESTINATION input)

add_test(btree btree "core" "2" "file" "input/mil.txt" "command" "input/command.txt")
