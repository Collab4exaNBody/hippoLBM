cmake_minimum_required(VERSION 3.22)

project(
  hippoLBM
  VERSION 0.0.1
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

## options

option(USE_AOS "Use Array Of Structure  data storage" OFF)
if(USE_AOS)
  message(STATUS "Use Array Of Structure data storage (Fi). Please do not use this option with GPU")
  add_compile_options(-DWFAOS)
endif()

## Onika
find_package(OpenMP REQUIRED)
find_package(onika REQUIRED)


# ======================================
# === add compiler option with CUDA  ===
# ======================================
if(ONIKA_BUILD_CUDA)
	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
endif()

add_subdirectory(src)

if(CMAKE_CXX_COMPILER_ID MATCHES Intel)
  set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} -Wall")
  set(CMAKE_CXX_FLAGS_DEBUG   "-g -traceback")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -xHOST")
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
  set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} -Wall")
  set(CMAKE_CXX_FLAGS_DEBUG   "-O0 -g3")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
endif()

set( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" "--expt-relaxed-constexpr" )

find_package(MPI REQUIRED)

add_executable(hippoLBM hippoLBM.cpp)
target_compile_options(hippoLBM PUBLIC ${ONIKA_COMPILE_OPTIONS})
target_compile_definitions(hippoLBM PUBLIC ${ONIKA_COMPILE_DEFINITIONS})
target_include_directories(hippoLBM PUBLIC ${ONIKA_INCLUDE_DIRS})
target_link_directories(hippoLBM PUBLIC ${ONIKA_LIBRARY_DIRS})
target_link_libraries(hippoLBM PUBLIC ${ONIKA_LIBRARIES})

## non-regression tests
add_subdirectory(regression-test)

## CREATE env.sh
if(DEFINED ONIKA_PLUGIN_PATH)
  list(APPEND ONIKA_PLUGIN_PATH ${CMAKE_CURRENT_BINARY_DIR}/plugins)
else()
  set(ONIKA_PLUGIN_PATH "${CMAKE_CURRENT_BINARY_DIR}/plugins")
endif()

string(REPLACE ";" ":" ONIKA_PLUGIN_PATH "${ONIKA_PLUGIN_PATH}")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/bin/setup-env.sh  
  "#!/bin/bash\n"
  "source ${onika_DIR}/bin/setup-env.sh\n"
	"export ONIKA_CONFIG_PATH=${CMAKE_CURRENT_SOURCE_DIR}/data/config\n"
	"export ONIKA_PLUGIN_PATH=${ONIKA_PLUGIN_PATH} \n")

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin/ DESTINATION bin )
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib/ DESTINATION bin )
