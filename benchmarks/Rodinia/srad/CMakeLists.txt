
set(SRC_FILES
  src/main.c
)

add_executable(srad ${SRC_FILES})
target_link_libraries(srad m)
