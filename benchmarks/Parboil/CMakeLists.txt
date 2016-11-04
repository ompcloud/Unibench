
include_directories(common)

add_library(parboilutil common/parboil.c)

add_custom_target(Parboil)

add_subdirectory(mri-q)
add_subdirectory(sgemm)
add_subdirectory(spmv)
add_subdirectory(stencil)
