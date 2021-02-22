#pragma once

#include "../utils.h"
#include "../IndexSet.h"
#include "VoronoiGraph.h"

class DelaunayApproximator {
public:
    using Polytope = VoronoiGraph::Polytope;

    DelaunayApproximator(int seed, const VoronoiGraph &graph, int delaunay_dim);

    void initialize_vertices();

    void random_walk(int n_steps, bool save_vertices = false);
    void full_walk();

    void save_delaunay(const std::string &output_filename);
    void save_voronoi(const std::string &output_filename);

    const VoronoiGraph &get_graph() const;

    bool should_compute_closure() const;

    void set_compute_closure(bool compute_closure);

private:
    RandomEngineMultithread re;

    const VoronoiGraph &graph;
    int n;
    int d;

    vec<Polytope> initial_vertex_vec;

    set_t<IndexSet> delaunay_skeleton;
    set_t<IndexSet> voronoi_vertices;

    int delaunay_dim = 2;
    bool compute_closure = true;

    static const int MAX_LOCAL_SIZE = 10000000;

    void add_delaunay_simplices(const IndexSet &simplex, set_t<IndexSet> &where_to);
};

