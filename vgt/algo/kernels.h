#pragma once

#include "../utils.h"
#include "../IndexSet.h"
#include "../KDTree.h"

enum RayStrategyType {
    BRUTE_FORCE = 0,
    BIN_SEARCH = 1,
    BRUTE_FORCE_GPU = 2
};

class Kernel {
public:
    enum IntersectionSearchType { // ray: // x(t) = x_0 + u * t
        RAY_INTERSECTION = 0,  // t > 0
        ANY_INTERSECTION = 1,  // t > 0 if the ray has an intersection, t < 0 otherwise
        LINE_INTERSECTION = 2  // both t > 0 and t < 0 (two intersections returned)
    };

    Kernel(const dmatrix &points);

    /**
     * Find an intersection of a ray with a voronoi boundary.
     * @param ref - the source of the ray
     * @param u - the direction of the ray (a unit vector)
     * @param j0 - an index of a data point closest to `ref`
     * @param ignore - indices of points that should be ignored, i.e. the generators of the current boundary
     *      (j0 should be included in the list)
     * @param best_j - an output parameter; the index of a new data point which would generate the next boundary,
     *      or -1 if the intersection with a data cell is not found;
     *      is two-dimensional when type==LINE_INTERSECTION
     * @param best_l - an output parameter; the length of the ray;
     *      may be negative if `may_ignore_direction` is set;
     *      will be a distance to the data bounds when best_j == -1;
     *      is two-dimensional when type==LINE_INTERSECTION
     * @param type
     *      RAY_INTERSECTION: intersection of the ray, `*best_l` is always non-negative
     *      ANY_INTERSECTION: intersection of the ray if exists, otherwise of the opposite ray
     *          (forces `*best_j != 1` almost surely, `*best_l` may be negative)
     *      LINE_INTERSECTION: both intersection of the line,
     *          both best_j and best_l should point to arrays of length two
     *
     */
     void intersect_ray(RayStrategyType strategy, const dvector &ref, const dvector &u, int j0, const IndexSet &ignore,
                        int *best_j, ftype *best_l, IntersectionSearchType type);

    /**
     * Find an intersection of a ray with a voronoi boundary, with a "brute force" strategy.
     */
    virtual void intersect_ray_bruteforce(const dvector &ref, const dvector &u, int j0, const IndexSet &ignore,
                                      int *best_j_arr, ftype *best_l_arr) const = 0;

    /**
     * Find an intersection of a ray with a voronoi boundary, with a "binary search" strategy.
     */
    virtual void intersect_ray_binsearch(const dvector &ref, const dvector &u, int j0, const IndexSet &ignore,
                                      int *best_j, ftype *best_l) const = 0;

    virtual int nearest_point(const dvector &ref, ftype *distance = nullptr) const;

    virtual int nearest_point_extra(const dvector &ref, ftype within_radius = -1,
                                       const IndexSet &ignore = {}, ftype *distance = nullptr) const = 0;

    virtual ftype distance(const dvector &a, const dvector &b) const = 0;

    virtual void project_to_tangent_space_inplace(const dvector &at, dvector &u) const = 0;

    virtual dvector move_along_ray(const dvector &start, const dvector &direction, ftype length) const = 0;

    virtual ftype max_length() const = 0;

    virtual ftype max_length(const const_dvector_ref &ref, const const_dvector_ref &u) const = 0;

    virtual void update_inserted_points(int n_new_points) = 0;
public:
    const dmatrix &points;
};

class Bounds {
public:
    virtual ftype max_length() const = 0;
    virtual ftype max_length(const const_dvector_ref &ref, const const_dvector_ref &u) const = 0;
    virtual bool contains(const const_dvector_ref &ref) const = 0;
};

class Unbounded : public Bounds {
public:
    ftype max_length() const override;
    ftype max_length(const const_dvector_ref &ref, const const_dvector_ref &u) const override;

    bool contains(const const_dvector_ref &ref) const override;
};

class BoundingBox : public Bounds {
public:
    BoundingBox(int dim, ftype half_width = 1);
    BoundingBox(const dvector &min, const dvector &max);
    ftype max_length() const override;
    ftype max_length(const const_dvector_ref &ref, const const_dvector_ref &u) const override;

    bool contains(const const_dvector_ref &ref) const override;

    const dvector &get_mn() const;
    const dvector &get_mx() const;

private:
    dvector mn, mx;
    ftype _max_length;
};

class BoundingSphere : public Bounds {
public:
    BoundingSphere(int dim, ftype radius = 1);
    BoundingSphere(const dvector &center, ftype radius);
    ftype max_length() const override;
    ftype max_length(const const_dvector_ref &ref, const const_dvector_ref &u) const override;

    bool contains(const const_dvector_ref &ref) const override;

    const dvector &get_center() const;

    ftype get_radius() const;

private:
    dvector center;
    ftype radius;
};

class EuclideanKernel : public Kernel {
public:
    EuclideanKernel(const dmatrix &points, const ptr<Bounds> &bounds = std::make_shared<Unbounded>());

    virtual void intersect_ray_bruteforce(const dvector &ref, const dvector &u, int j0, const IndexSet &ignore,
                                          int *best_j_arr, ftype *best_l_arr) const override;

    void intersect_ray_precomputed(const const_dvector_ref &ref, const const_dvector_ref &u, int j0, int j, ftype *length) const;

    virtual void intersect_ray_binsearch(const dvector &ref, const dvector &u, int j0, const IndexSet &ignore,
                                         int *best_j, ftype *best_l) const override;

    virtual int nearest_point_extra(const dvector &ref, ftype within_radius = -1,
                                       const IndexSet &ignore = {}, ftype *distance = nullptr) const override;

    virtual ftype distance(const dvector &a, const dvector &b) const override;

    virtual void project_to_tangent_space_inplace(const dvector &at, dvector &u) const override;

    virtual dvector move_along_ray(const dvector &start, const dvector &direction, ftype length) const override;

    virtual ftype max_length() const override;

    virtual ftype max_length(const const_dvector_ref &ref, const const_dvector_ref &u) const override;

    void precompute(const dmatrix &ref_mat, const dmatrix &u_mat);

    void reset_ref_mat(int ref_n);

    void reset_u_mat(int u_n);

    const ptr<Bounds> &get_bounds() const;

    virtual void update_inserted_points(int n_new_points) override;

protected:
    KDTree tree;
    ptr<Bounds> bounds;

public:
    ftype MAX_RAY_LENGTH = 1000000;
    ftype STARTING_LENGTH_UPDATE_ALPHA = static_cast<ftype>(0.5);
    int MAX_BIN_SEARCH_ITER = 30;
    ftype BIN_SEARCH_SPLIT_PARAMETER = static_cast<ftype>(0.5);
    ftype BIN_SEARCH_MIN_INTERVAL = static_cast<ftype>(1e-7);

    // precomputations
    mutable dynmatrix precomputed_u_dot_data;
    mutable dynmatrix precomputed_ref_dot_data;
    mutable dynvector precomputed_data_squared;
    mutable dynvector precomputed_ref_squared;
    mutable dynmatrix precomputed_u_dot_ref;
};

class SphericalKernel : public Kernel {
public:
    SphericalKernel(const dmatrix &points);

    virtual void intersect_ray_bruteforce(const dvector &ref, const dvector &u, int j0, const IndexSet &ignore,
                                          int *best_j_arr, ftype *best_l_arr) const override;

    virtual void intersect_ray_binsearch(const dvector &ref, const dvector &u, int j0, const IndexSet &ignore,
                                         int *best_j, ftype *best_l) const override;

    virtual int nearest_point_extra(const dvector &ref, ftype within_radius = -1,
                                       const IndexSet &ignore = {}, ftype *distance = nullptr) const override;

    virtual ftype distance(const dvector &a, const dvector &b) const override;

    virtual void project_to_tangent_space_inplace(const dvector &at, dvector &u) const override;

    virtual dvector move_along_ray(const dvector &start, const dvector &direction, ftype length) const override;

    virtual ftype max_length() const override;

    virtual ftype max_length(const const_dvector_ref &ref, const const_dvector_ref &u) const override;

    void update_inserted_points(int n_new_points) override;

protected:
    KDTree tree;

public:
    float MAX_RAY_LENGTH = 1000;  // todo adjust to data...
    int MAX_BIN_SEARCH_ITER = 30;
    ftype BIN_SEARCH_SPLIT_PARAMETER = static_cast<ftype>(0.5);
    ftype BIN_SEARCH_MIN_INTERVAL = static_cast<ftype>(1e-7);
};
