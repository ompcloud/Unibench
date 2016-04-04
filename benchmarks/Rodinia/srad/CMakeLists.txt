
set(SRC_DIR src)

set(SRC_FILES
  ${SRC_DIR}/main.c
)

add_executable(srad ${SRC_FILES})
target_link_libraries(srad m)
add_dependencies(Rodinia srad)

add_test(Rodinia_srad srad 1000 0.5 1024 1024 4)
