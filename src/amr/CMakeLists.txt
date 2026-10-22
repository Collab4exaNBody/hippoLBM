set(hippo_amr_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/include)
set(hippo_amr_LINK_LIBRARIES hippo_grid hippo_collision hippo_io hippo_bcs hippo_object)
hippolbm_add_plugin(hippo_amr ${CMAKE_CURRENT_SOURCE_DIR})
