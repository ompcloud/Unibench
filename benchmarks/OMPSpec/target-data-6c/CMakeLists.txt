
set(SRC_FILES
  src/target-data-6c.c
)

add_executable(target-data-6c ${SRC_FILES})

add_test(OMPSpec_target-data-6c target-data-6c)
