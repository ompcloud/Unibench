
set(SRC_FILES
  src/target-3c.c
)

add_executable(target-3c ${SRC_FILES})

add_test(OMPSpec_target-3c target-3c)
