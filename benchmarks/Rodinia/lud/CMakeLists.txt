
set(SRC_DIR src)

set(SRC_FILES
  ${SRC_DIR}/lud.c
  ${SRC_DIR}/lud_omp.c
  ${SRC_DIR}/common/common.c
)

add_executable(lud ${SRC_FILES})
target_link_libraries(lud m)
add_dependencies(Rodinia lud)

#add_test(Rodinia_lud lud "-s" "1024" "-v")
