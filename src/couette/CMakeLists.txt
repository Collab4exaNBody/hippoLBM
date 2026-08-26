set(hippo_couette_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/include)
set(hippo_couette_LINK_LIBRARIES hippo_grid hippo_collision)
hippolbm_add_plugin(hippo_couette ${CMAKE_CURRENT_SOURCE_DIR})
