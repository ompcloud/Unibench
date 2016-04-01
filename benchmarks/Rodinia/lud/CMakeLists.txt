
set(SRC_FILES
  src/lud.c
  src/lud_omp.c
  src/common/common.c
)

add_executable(lud ${SRC_FILES})
target_link_libraries(lud m)
