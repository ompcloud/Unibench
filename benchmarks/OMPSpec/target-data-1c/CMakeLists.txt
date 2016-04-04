
set(SRC_FILES
  src/target-data-1c.c
)

add_executable(target-data-1c ${SRC_FILES})

add_test(OMPSpec_target-data-1c target-data-1c)
