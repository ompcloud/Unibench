
set(SRC_DIR src)

set(SRC_FILES
  ${SRC_DIR}/target-data-6c.c
)

add_executable(target-data-6c ${SRC_FILES})
add_dependencies(OMPSpec target-data-6c)

add_test(OMPSpec_target-data-6c target-data-6c)
