
include_directories(common)

add_custom_target(mgBench)

add_subdirectory(cholesky)
add_subdirectory(collinear-list)
add_subdirectory(floyd)
add_subdirectory(k-nearest)
add_subdirectory(lu-decomposition)
add_subdirectory(mat-mul)
add_subdirectory(other-nearest)
add_subdirectory(search-vector)
add_subdirectory(str-matching)
add_subdirectory(vector-product)
