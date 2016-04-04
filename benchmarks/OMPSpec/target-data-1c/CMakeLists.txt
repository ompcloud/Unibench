
set(SRC_DIR src)

set(SRC_FILES
  ${SRC_DIR}/target-data-1c.c
)

add_executable(target-data-1c ${SRC_FILES})
add_dependencies(OMPSpec target-data-1c)

add_test(OMPSpec_target-data-1c target-data-1c)
