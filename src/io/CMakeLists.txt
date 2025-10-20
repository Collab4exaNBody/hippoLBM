set(hippo_io_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/include)
set(hippo_io_LINK_LIBRARIES hippo_grid)
hippolbm_add_plugin(hippo_io ${CMAKE_CURRENT_SOURCE_DIR})
