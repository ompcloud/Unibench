
set(SRC_DIR src)

set(SRC_FILES
  ${SRC_DIR}/target-3c.c
)

add_executable(target-3c ${SRC_FILES})
add_dependencies(OMPSpec target-3c)

add_test(OMPSpec_target-3c target-3c)
