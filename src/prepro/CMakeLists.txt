set(hippo_prepro_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/include)
set(hippo_prepro_LINK_LIBRARIES hippo_grid)
hippolbm_add_plugin(hippo_prepro ${CMAKE_CURRENT_SOURCE_DIR})
