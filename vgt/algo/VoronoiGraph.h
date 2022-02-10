#pragma once

#include "../utils.h"
#include "../RandomEngine.h"
#include "../KDTree.h"
#include "kernels.h"

enum DataType {
    EUCLIDEAN = 0,
    TOROIDAL = 1,
    SPHERICAL = 2
};

class VoronoiGraph {
public:
    struct Polytope {
        Polytope();
        Polytope(const IndexSet &dual, const dvector &ref);
        bool is_none() const;

        IndexSet dual;
        dvector ref;
    };

    static const Polytope NONE;

    explicit VoronoiGraph(RayStrategyType strategy = BRUTE_FORCE, DataType data_type = EUCLIDEAN);

    /**
     * Check whether the polytope is a Voronoi vertex, i.e. is 0-dim.
     */
    bool is_vertex(const Polytope &p) const;

    void initialize(const dmatrix &points, ptr<Bounds> bounds = nullptr);

    /**
     * Read data from a .npy file. Also initializes the data kernel.
     * @param filename
     */
    void read_points(const std::string &filename, ptr<Bounds> bounds = nullptr);

    void insert(const dmatrix &new_points);

    /**
     * Perform a descent to obtain a random Voronoi vertex
     * of a Voronoi cell of the data point.
     * Returns NONE if didn't succeed (e.g. a numeric precision issue).
     * @param point_idx id of the data point
     * @param re random engine
     */
    Polytope retrieve_vertex_nearby(int point_idx, RandomEngine &re) const;

    /**
     * Returns the index of a generator of a Voronoi cell that contains the given point.
     * Equivalent to the nearest neighbor lookup.
     */
    int get_containing_voronoi_cell(const dvector &ref) const;

    /**
     * Perform a descent to obtain a random Voronoi vertex
     * of a Voronoi cell containing the reference point.
     * Returns NONE if didn't succeed (e.g. a numeric precision issue).
     * @param ref reference point
     * @param re random engine
     * @param nearest_idx id of the nearest data point in data space, -1 for an unknown nearest neighbor
     */
    Polytope retrieve_vertex_nearby(const dvector &ref, RandomEngine &re, int nearest_idx = -1) const;

    vec<Polytope> retrieve_vertices_nearby(const dmatrix &ref_mat, RandomEngineMultithread &re, vec<int> nearest_idx_vec) const;

    /**
     * Perform a visibility walk from a random nearby Voronoi vertex
     * to obtain another vertex, the dual of which (a Delaunay d-simplex)
     * contains the given point.
     * Returns NONE if the point is on the outside of the convex hull of the data.
     * @param coordinates (optional) returns barycentric coordinates of the point in the simplex
     */
    Polytope retrieve_vertex(const dvector &point, RandomEngine &re, svector *coordinates = nullptr) const;

    vec<Polytope> retrieve_vertices(const dmatrix &queries, RandomEngineMultithread &re,
                                    vec<svector> *coordinates = nullptr) const;

    /**
     * Returns a neighbor of a vertex in the Voronoi graph at a given index.
     * Returns NONE is the edge is infinite at that index.
     */
    Polytope get_neighbor(const Polytope &v, int index, RandomEngine &re) const;

    /**
     * Returns all neighbors of a vertex in the Voronoi graph.
     * The vector has size (D+1) and will contain NONE for infinite edges.
     */
    vec<Polytope> get_neighbors(const Polytope &v, RandomEngine &re) const;

    vec<vec<Polytope>> get_neighbors(const vec<Polytope> &vertices, RandomEngineMultithread &re) const;

    vec<Polytope> get_neighbors(const vec<Polytope> &vertices, const vec<int> &indices,
                                RandomEngineMultithread &re) const;

    /**
     * Samples a point uniformly on a (k-1)-sphere within the provided
     * k-dimensional polytope and cast a ray in that direction from
     * the reference point of the polytope. The method returns a face
     * of the polytope, or NONE in case of an infinite direction.
     *
     * `direction` can be given as a `re.rand_on_sphere()` and does not have
     * to lie within the polytope (though, it should not be strictly orthogonal to it).
     *
     * Orthogonal complement is an optional parameter, and will be computed
     * if not provided.
     *
     * Using `bidirectional=true` together with `strategy=brute_force` guarantees
     * with probability=1 that a non-NONE polytope will be returned.
     * `bidirectional` is ignored when `strategy=bin_search`.
     *
     * 'length' is an optional output parameter to return the length of the ray until the
     * intersection. Even if the intersection is NONE, the length will still be returned
     * as the distance to the intersection with manifold boundary.
     */
    Polytope cast_ray(const Polytope &p, const dvector &direction,
                      ptr<const vec<dvector>> orthogonal_complement = nullptr, bool bidirectional = false,
                      ftype *length = nullptr) const;

    const dmatrix& get_data() const;

    int get_data_size() const;

    int get_data_dim() const;

    /**
     * Resets the number of successful and failed validations.
     */
    void reset_failure_rate_counters() const;

    /**
     * Return a pair, where `first` is the number of successful validations,
     * and `second` is the number of failed validations.
     *
     * A validation is run for each new Polytope and a corresponding reference point.
     */
    std::pair<long long, long long> get_failure_rate() const;

    void print_validations_info() const;

    Kernel& get_kernel() const;

    RayStrategyType get_strategy() const;

    DataType get_data_type() const;

private:
    RayStrategyType strategy;
    DataType data_type;
    mutable ptr<Kernel> kernel;

    int n_points = 0;
    int ambient_dim = 0;
    int data_dim = 0;
    dmatrix points;

    mutable long long validations_failed = 0;
    mutable long long validations_ok = 0;

    bool validate(const vec<int> &equidistants, const dvector &ref) const;

public:
    const ftype VALIDATION_EPS = static_cast<ftype>(1e-5);

    const int MAX_RETRIES = 5;

    int VISIBILITY_WALK_MAX_STEPS = 10000000;
};
