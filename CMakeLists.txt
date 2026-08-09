cmake_minimum_required(VERSION 3.22)

project(
  hippoLBM
  VERSION 1.0.0
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ===========================
# === CMake customization ===
# ===========================
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Release RelWithDebInfo Debug)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

## setup hippolbm-config.cmake
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
message(STATUS "module " ${CMAKE_MODULE_PATH})

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/hippolbm-config-step1.cmake")
include(${CMAKE_MODULE_PATH}/hippolbm_add_plugins.cmake)

## options
option(USE_AOS "Use Array Of Structure  data storage" OFF)
if(USE_AOS)
  message(STATUS "Use Array Of Structure data storage (Fi). Please do not use this option with GPU")
  add_compile_options(-DWFAOS)
endif()

## Onika
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
find_package(onika REQUIRED)
find_package(ZLIB REQUIRED)


# ======================================
# === add compiler option with CUDA  ===
# ======================================
if(ONIKA_BUILD_CUDA)
	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
endif()

add_subdirectory(src)

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/hippolbm-config-step2.cmake")

# ===========================================================
# === generate execution wrappers and environment scripts ===
# ===========================================================

if(DEFINED ONIKA_PLUGIN_PATH)
  list(APPEND ONIKA_PLUGIN_PATH ${CMAKE_CURRENT_BINARY_DIR}/plugins)
else()
  set(ONIKA_PLUGIN_PATH "${onika_DIR}/plugins;${CMAKE_CURRENT_BINARY_DIR}/plugins")
endif()
string(REPLACE ";" ":" ONIKA_PLUGIN_PATH "${ONIKA_PLUGIN_PATH}")

## post-install environment setup script, sourced by the installed wrapper
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
file(WRITE  ${CMAKE_CURRENT_BINARY_DIR}/bin/setup-env.sh
  "#!/bin/bash\n"
  "source ${onika_DIR}/bin/setup-env.sh\n"
  "export ONIKA_CONFIG_PATH=${CMAKE_INSTALL_PREFIX}/data/config\n"
  "export ONIKA_PLUGIN_PATH=${onika_DIR}/plugins:${CMAKE_INSTALL_PREFIX}/plugins\n"
  "[ $ONIKA_DEBUG ] && echo \"hippoLBM environment : LD_LIBRARY_PATH=$LD_LIBRARY_PATH , ONIKA_CONFIG_PATH=$ONIKA_CONFIG_PATH , ONIKA_PLUGIN_PATH=$ONIKA_PLUGIN_PATH , ONIKA_DATA_PATH=$ONIKA_DATA_PATH\"\n")

## post-install execution wrapper (installed to <prefix>/bin/hippoLBM)
file(WRITE  ${CMAKE_CURRENT_BINARY_DIR}/bin/hippoLBM
  "#!/bin/bash\n"
  "# hippoLBM execution wrapper\n"
  "source ${CMAKE_INSTALL_PREFIX}/bin/setup-env.sh\n"
  "${ONIKA_EXEC} $*\n")

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin/ DESTINATION bin
  FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib/ DESTINATION lib )
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/hippoLBM DESTINATION include)

## pre-installation execution wrapper, to run hippoLBM directly from the build directory
file(WRITE  ${CMAKE_CURRENT_BINARY_DIR}/hippoLBM
  "#!/bin/bash\n"
  "# Pre-installation hippoLBM execution wraper\n"
  "source ${onika_DIR}/bin/setup-env.sh\n"
  "export LD_LIBRARY_PATH=${CMAKE_CURRENT_BINARY_DIR}/lib:$LD_LIBRARY_PATH\n"
  "export ONIKA_CONFIG_PATH=${CMAKE_CURRENT_SOURCE_DIR}/data/config\n"
  "export ONIKA_PLUGIN_PATH=${ONIKA_PLUGIN_PATH}\n"
  "[ $ONIKA_DEBUG ] && echo \"hippoLBM pre-install environment : LD_LIBRARY_PATH=$LD_LIBRARY_PATH , ONIKA_CONFIG_PATH=$ONIKA_CONFIG_PATH , ONIKA_PLUGIN_PATH=$ONIKA_PLUGIN_PATH , ONIKA_DATA_PATH=$ONIKA_DATA_PATH\"\n"
  "${ONIKA_EXEC} $*\n")
file(CHMOD ${CMAKE_CURRENT_BINARY_DIR}/hippoLBM FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

## non-regression tests (rely on the pre-install wrapper above)
add_subdirectory(regression-test)
