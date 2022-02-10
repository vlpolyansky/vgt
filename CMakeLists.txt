cmake_minimum_required(VERSION 3.6)
project(voronoi_approximator)

set(CMAKE_BUILD_TYPE "CondaDebug")
set(CMAKE_CXX_STANDARD 17)

# find_package(CGAL REQUIRED COMPONENTS Core)
# include( ${CGAL_USE_FILE} )
# ADD_DEFINITIONS("-DCGAL_DISABLE_ROUNDING_MATH_CHECK -frounding-math")

find_package(Eigen3)
message("-- Found Eigen use file: ${EIGEN3_USE_FILE}")
include( ${EIGEN3_USE_FILE} )

add_subdirectory(libs)

include_directories(${CMAKE_SOURCE_DIR}/include)

INCLUDE(CheckCXXSourceCompiles)

#ADD_DEFINITIONS("-DCGAL_DISABLE_ROUNDING_MATH_CHECK -frounding-math")

FIND_PACKAGE(OpenMP)

set (CMAKE_REQUIRED_FLAGS ${OpenMP_CXX_FLAGS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -frounding-math -O3")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O3")


add_subdirectory(vgt)
