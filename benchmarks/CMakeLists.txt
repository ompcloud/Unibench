
add_custom_target(experiments)

include_directories(common)

add_subdirectory(mgBench)
add_subdirectory(OMPSpec)
add_subdirectory(kastors)
add_subdirectory(Parboil)
add_subdirectory(Polybench)
add_subdirectory(Rodinia)
