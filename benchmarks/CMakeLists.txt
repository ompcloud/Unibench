
add_custom_target(experiments)

include_directories(common)

# these are not ready for offloading. no target keyword
#add_subdirectory(BOTS)
#add_subdirectory(kastors)
#add_subdirectory(toysreal)
add_subdirectory(mgBench)
add_subdirectory(OMPSpec)
add_subdirectory(Parboil)
add_subdirectory(Polybench)
add_subdirectory(Rodinia)
