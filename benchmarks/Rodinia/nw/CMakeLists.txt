
set(SRC_DIR src)

set(SRC_FILES
  ${SRC_DIR}/needle.cpp
)

add_executable(nw ${SRC_FILES})
add_dependencies(Rodinia nw)

#add_test(Rodinia_nw nw 2048 10 2)
