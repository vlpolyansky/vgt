set(SOURCES utils.cpp utils.h RandomEngine.cpp RandomEngine.h
        IndexSet.cpp IndexSet.h KDTree.cpp KDTree.h)


add_executable(approx_delaunay algo/DelaunayApproximator.cpp algo/DelaunayApproximator.h
        algo/VoronoiGraph.cpp algo/VoronoiGraph.h ${SOURCES})
target_link_libraries(approx_delaunay cnpy)

add_executable(interpolate algo/interpolate.cpp algo/VoronoiGraph.cpp algo/VoronoiGraph.h ${SOURCES})
target_link_libraries(interpolate cnpy)

add_executable(prepare_patches extra/prepare_patches.cpp ${SOURCES})
target_link_libraries(prepare_patches cnpy)

set_target_properties(prepare_patches PROPERTIES EXCLUDE_FROM_ALL TRUE)

#
#add_executable(bottleneck_distance va/_bottleneck_distance.cpp ${SOURCES} va/ph_tools.cpp va/ph_tools.h upd/VoronoiGraph.cpp upd/VoronoiGraph.h upd/DelaunayApproximator.cpp upd/DelaunayApproximator.h)
#target_link_libraries(bottleneck_distance cnpy)
#
#add_executable(compute_persistence va/_compute_persistence.cpp ${SOURCES} va/ph_tools.cpp va/ph_tools.h va/VoronoiSampler.cpp va/VoronoiSampler.h)
#target_link_libraries(compute_persistence cnpy)
#
#add_executable(calc_difference test/_calc_difference.cpp ${SOURCES} va/ph_tools.cpp va/ph_tools.h)
#target_link_libraries(calc_difference cnpy)
#