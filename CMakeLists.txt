cmake_minimum_required(VERSION 3.6)
project(voronoi_approximator)

set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_STANDARD 17)

# find_package(CGAL REQUIRED COMPONENTS Core)
# include( ${CGAL_USE_FILE} )
# ADD_DEFINITIONS("-DCGAL_DISABLE_ROUNDING_MATH_CHECK -frounding-math")

find_package(Eigen3)
include( ${EIGEN3_USE_FILE} )

#set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR})

add_subdirectory(libs)

include_directories(${CMAKE_SOURCE_DIR}/include)

INCLUDE(CheckCXXSourceCompiles)

#ADD_DEFINITIONS("-DCGAL_DISABLE_ROUNDING_MATH_CHECK -frounding-math")

FIND_PACKAGE(OpenMP)

set (CMAKE_REQUIRED_FLAGS ${OpenMP_CXX_FLAGS})
CHECK_CXX_SOURCE_COMPILES("
#include <omp.h>
int main() {
#if (_OPENMP >= 200805 || _MSC_VER >= 1500)
  return 0;
#else
  breaks_on_purpose
#endif
}
" OPENMP_VERSION)

if(OPENMP_VERSION)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
    message("

OpenMP 3.0 not supported by the compiler (${CMAKE_CXX_COMPILER})!
To get optimal performance for the \"chunk\" algorithm, use a C++ compiler with OpenMP 3.0 support (e.g., GCC >=4.4).
To use a different compiler, pass it to cmake in the variable CMAKE_CXX_COMPILER:
 cmake . -DCMAKE_CXX_COMPILER=g++-4.7

")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -frounding-math -O3")

#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O3")


add_subdirectory(vgt)
