
set(SRC_DIR src)

set(SRC_FILES
  ${SRC_DIR}/target-data-2c.c
)

add_executable(target-data-2c ${SRC_FILES})
add_dependencies(OMPSpec target-data-2c)

add_test(OMPSpec_target-data-2c target-data-2c)
