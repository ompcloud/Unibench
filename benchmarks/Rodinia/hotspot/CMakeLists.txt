
set(SRC_DIR src)

set(SRC_FILES
  ${SRC_DIR}/hotspot_openmp.cpp
)

add_executable(hotspot ${SRC_FILES})
add_dependencies(Rodinia hotspot)

file(COPY input DESTINATION .)

add_test(Rodinia_hotspot hotspot 512 512 2 4 "input/temp_512" "input/power_512")
