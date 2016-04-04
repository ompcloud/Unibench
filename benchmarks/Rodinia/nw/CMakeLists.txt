
set(SRC_FILES
  src/needle.cpp
)

add_executable(nw ${SRC_FILES})

add_test(nw nw 2048 10 2)
