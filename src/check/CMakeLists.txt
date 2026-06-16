set(hippo_check_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/include)
set(hippo_check_LINK_LIBRARIES hippo_grid hippo_io)
hippolbm_add_plugin(hippo_check ${CMAKE_CURRENT_SOURCE_DIR})
