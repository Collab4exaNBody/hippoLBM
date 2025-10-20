set(hippo_bcs_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/include)
set(hippo_bcs_LINK_LIBRARIES hippo_grid)
hippolbm_add_plugin(hippo_bcs ${CMAKE_CURRENT_SOURCE_DIR})
