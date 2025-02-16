cmake_minimum_required(VERSION 3.22)

project(
  hipoLBM
  VERSION 0.0.1
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## options

option(USE_AOS "Use Array Of Structure  data storage" OFF)
if(USE_AOS)
  message(STATUS "Use Array Of Structure data storage (Fi). Please do not use this option with GPU")
  add_compile_options(-DWFAOS)
endif()


## Onika
find_package(OpenMP REQUIRED)
find_package(onika REQUIRED)

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

add_executable(hipoLBM hipoLBM.cpp)
target_compile_options(hipoLBM PUBLIC ${ONIKA_COMPILE_OPTIONS})
target_compile_definitions(hipoLBM PUBLIC ${ONIKA_COMPILE_DEFINITIONS})
target_include_directories(hipoLBM PUBLIC ${ONIKA_INCLUDE_DIRS})
target_link_directories(hipoLBM PUBLIC ${ONIKA_LIBRARY_DIRS})
target_link_libraries(hipoLBM PUBLIC ${ONIKA_LIBRARIES})
