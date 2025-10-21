set(hippo_collision_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/include)
set(hippo_collision_LINK_LIBRARIES hippo_grid)
hippolbm_add_plugin(hippo_collision ${CMAKE_CURRENT_SOURCE_DIR})
