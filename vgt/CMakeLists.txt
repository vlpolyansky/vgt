set(SOURCES utils.cpp utils.h RandomEngine.cpp RandomEngine.h
        IndexSet.cpp IndexSet.h KDTree.cpp KDTree.h)

set(VG_SOURCES algo/VoronoiGraph.cpp algo/VoronoiGraph.h algo/kernels.cpp algo/kernels.h
        algo/kernels_gpu.cpp algo/kernels_gpu.h cl/voronoi.cl cl/voronoi_cl.h
        ${SOURCES})

convertIntoHeader(cl/voronoi.cl cl/voronoi_cl.h voronoi_kernel_sources)

link_libraries(cnpy)

add_executable(approx_delaunay algo/DelaunayApproximator.cpp algo/DelaunayApproximator.h ${VG_SOURCES})
target_link_libraries(approx_delaunay libclew libgpu ${Boost_LIBRARIES})

add_executable(significant_delaunay main/significant_delaunay.cpp ${VG_SOURCES})
target_link_libraries(significant_delaunay libclew libgpu)

add_executable(interpolate algo/interpolate.cpp algo/VoronoiGraph.cpp algo/VoronoiGraph.h ${SOURCES})
target_link_libraries(interpolate cnpy)

add_executable(prepare_patches extra/prepare_patches.cpp ${SOURCES})
target_link_libraries(prepare_patches cnpy)
set_target_properties(prepare_patches PROPERTIES EXCLUDE_FROM_ALL TRUE)
