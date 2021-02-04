
include_directories(common)

add_custom_target(kastors)

add_subdirectory(jacobi)
add_subdirectory(plasma)
add_subdirectory(sparselu)
add_subdirectory(strassen)
