cmake_minimum_required(VERSION 3.22)

project(
  hipoLBM
  VERSION 0.0.1
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## Onika
find_package(onika REQUIRED)

add_subdirectory(src)
#add_subdirectory(data)

add_executable(hipoLBM hipoLBM.cpp)
target_compile_options(hipoLBM PUBLIC ${ONIKA_COMPILE_OPTIONS})
target_compile_definitions(hipoLBM PUBLIC ${ONIKA_COMPILE_DEFINITIONS})
target_include_directories(hipoLBM PUBLIC ${ONIKA_INCLUDE_DIRS})
target_link_directories(hipoLBM PUBLIC ${ONIKA_LIBRARY_DIRS})
target_link_libraries(hipoLBM PUBLIC ${ONIKA_LIBRARIES})
