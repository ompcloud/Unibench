
include_directories(common)

add_library(parboil common/parboil.c)

add_subdirectory(mri-q)
add_subdirectory(sgemm)
add_subdirectory(spmv)
add_subdirectory(stencil)
