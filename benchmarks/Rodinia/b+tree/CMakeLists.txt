
set(SRC_DIR src)

set(SRC_FILES
  ${SRC_DIR}/main.c
  ${SRC_DIR}/util/num/num.c
  ${SRC_DIR}/util/timer/timer.c
  ${SRC_DIR}/kernel/kernel_cpu.c
  ${SRC_DIR}/kernel/kernel_cpu_2.c
)

add_executable(b+tree ${SRC_FILES})
target_link_libraries(b+tree m)

file(COPY input DESTINATION input)

add_test(Rodinia_b+tree b+tree "core" "2" "file" "input/mil.txt" "command" "input/command.txt")
