
#include "kernels.h"

#define UNNAN(precomputed, value, expression) ((precomputed) ? (math::isnan(value) ? value = (expression) : value) : (expression))

Kernel::Kernel(const dmatrix &points) : points(points) {}

int Kernel::nearest_point(const dvector &ref, ftype *distance) const {
    return nearest_point_extra(ref, -1, {}, distance);
}

void Kernel::intersect_ray(RayStrategyType strategy,
                           const dvector &ref, const dvector &u, int j0, const IndexSet &ignore,
                           int *best_j, ftype *best_l, IntersectionSearchType type) {
    if (strategy == BRUTE_FORCE || strategy == BRUTE_FORCE_GPU) {
        if (strategy == BRUTE_FORCE_GPU) {
            static bool warning_shown = false;
            if (!warning_shown) {
                std::cout << "Warning: CPU ray intersection while the strategy is GPU. Intended?" << std::endl;
                warning_shown = true;
            }
        }

        int tmp_j[2];
        ftype tmp_l[2];
        intersect_ray_bruteforce(ref, u, j0, ignore, tmp_j, tmp_l);

        switch (type) {
            case RAY_INTERSECTION:
                *best_j = tmp_j[0];
                *best_l = tmp_l[0];
                break;
            case ANY_INTERSECTION: {
                int picked = 0;
                if (tmp_j[0] < 0) {
                    picked = 1;
                }
                *best_j = tmp_j[picked];
                *best_l = tmp_l[picked];
                break;
            }
            case LINE_INTERSECTION:
                best_j[0] = tmp_j[0]; best_j[1] = tmp_j[1];
                best_l[0] = tmp_l[0]; best_l[1] = tmp_l[1];
                break;
        }
    } else if (strategy == BIN_SEARCH) {
        intersect_ray_binsearch(ref, u, j0, ignore, best_j, best_l);
        switch (type) {
            case RAY_INTERSECTION:
                break;
            case ANY_INTERSECTION:
                if (*best_j == -1) {
                    intersect_ray_binsearch(ref, -u, j0, ignore, best_j, best_l);
                    *best_l = -(*best_l);
                }
                break;
            case LINE_INTERSECTION:
                intersect_ray_binsearch(ref, -u, j0, ignore, best_j + 1, best_l + 1);
                break;
        }
    } else {
        throw std::runtime_error("Unknown strategy: " + std::to_string(strategy));
    }
}

EuclideanKernel::EuclideanKernel(const dmatrix &points, const ptr<Bounds> &bounds) :
        Kernel(points), tree(points), bounds(bounds) {
    int data_n = points.cols();
    precomputed_data_squared = dynvector::Constant(data_n, NAN);
    tree.init();
}

SphericalKernel::SphericalKernel(const dmatrix &points) : Kernel(points), tree(points) {
    tree.init();
}

ftype EuclideanKernel::max_length() const {
    return bounds->max_length();
}

ftype SphericalKernel::max_length() const {
    return PI_ftype;
}

ftype EuclideanKernel::max_length(const const_dvector_ref &ref, const const_dvector_ref &u) const {
    return bounds->max_length(ref, u);
}

ftype SphericalKernel::max_length(const const_dvector_ref &ref, const const_dvector_ref &u) const {
    return PI_ftype;
}

void EuclideanKernel::intersect_ray_bruteforce(const dvector &ref, const dvector &u, int j0, const IndexSet &ignore,
                                               int *best_j_arr, ftype *best_l_arr) const {
    best_j_arr[0] = best_j_arr[1] = -1;
    best_l_arr[0] = max_length(ref, u);
    best_l_arr[1] = -max_length(ref, -u);
    int n_points = points.cols();

    ftype p0_squared;
    ftype u_dot_p0;
    ftype ref_dot_p0;
    if (j0 >= 0) {
        p0_squared = UNNAN(true,
                                 precomputed_data_squared(j0),
                                 points.col(j0).squaredNorm());
        u_dot_p0 = UNNAN(u.index >= 0,
                               precomputed_u_dot_data(u.index, j0),
                               u.dot(points.col(j0)));
        ref_dot_p0 = UNNAN(ref.index >= 0,
                                 precomputed_ref_dot_data(ref.index, j0),
                                 ref.dot(points.col(j0)));
    } else {   // INSERTION CELL !!!
        p0_squared = UNNAN(ref.index >= 0,
                           precomputed_ref_squared(ref.index),
                           ref.squaredNorm());
        u_dot_p0 = UNNAN(u.index >= 0 && ref.index >= 0,
                         precomputed_u_dot_ref(u.index, ref.index),
                         u.dot(ref));
        ref_dot_p0 = p0_squared;
    }
    // Go through all other points: O(n)
    for (int j = 0; j < n_points; j++) {
        if (ignore.contains(j)) {
            continue;
        }
        ftype pj_squared = UNNAN(true,
                                 precomputed_data_squared(j),
                                 points.col(j).squaredNorm());
        ftype u_dot_pj = UNNAN(u.index >= 0,
                               precomputed_u_dot_data(u.index, j),
                               u.dot(points.col(j)));
        ftype ref_dot_pj = UNNAN(ref.index >= 0,
                                 precomputed_ref_dot_data(ref.index, j),
                                 ref.dot(points.col(j)));
        ftype cur_l = (p0_squared - pj_squared - 2 * (ref_dot_p0 - ref_dot_pj)) /
                      (2 * (u_dot_p0 - u_dot_pj));
//        if (cur_l >= 0 && (cur_l < best_l_arr[0])) {
        if (u_dot_pj - u_dot_p0 > 0 && (cur_l < best_l_arr[0])) { // direction towards pj
            best_j_arr[0] = j;
            best_l_arr[0] = cur_l;
        }
//        if (cur_l <= 0 && (cur_l > best_l_arr[1])) {
        if (u_dot_pj - u_dot_p0 < 0 && (cur_l > best_l_arr[1])) { // direction from pj
            best_j_arr[1] = j;
            best_l_arr[1] = cur_l;
        }
    }
}

void
EuclideanKernel::intersect_ray_precomputed(const const_dvector_ref &ref, const const_dvector_ref &u, int j0, int j, ftype *length) const {
    if (j < 0) {
        ensure(j != -2, "Error occurred in ray casting!");
        *length = max_length(ref, u);
        return;
    }
    ftype p0_squared;
    ftype u_dot_p0;
    ftype ref_dot_p0;
    if (j0 >= 0) {
        p0_squared = UNNAN(true,
                           precomputed_data_squared(j0),
                           points.col(j0).squaredNorm());
        u_dot_p0 = UNNAN(u.index >= 0,
                         precomputed_u_dot_data(u.index, j0),
                         u.dot(points.col(j0)));
        ref_dot_p0 = UNNAN(ref.index >= 0,
                           precomputed_ref_dot_data(ref.index, j0),
                           ref.dot(points.col(j0)));
    } else {   // INSERTION CELL !!!
        p0_squared = UNNAN(ref.index >= 0,
                           precomputed_ref_squared(ref.index),
                           ref.squaredNorm());
        u_dot_p0 = UNNAN(u.index >= 0 && ref.index >= 0,
                         precomputed_u_dot_ref(u.index, ref.index),
                         u.dot(ref));
        ref_dot_p0 = p0_squared;
    }
    ftype pj_squared = UNNAN(true,
                             precomputed_data_squared(j),
                             points.col(j).squaredNorm());
    ftype u_dot_pj = UNNAN(u.index >= 0,
                           precomputed_u_dot_data(u.index, j),
                           u.dot(points.col(j)));
    ftype ref_dot_pj = UNNAN(ref.index >= 0,
                             precomputed_ref_dot_data(ref.index, j),
                             ref.dot(points.col(j)));
    ftype cur_l = (p0_squared - pj_squared - 2 * (ref_dot_p0 - ref_dot_pj)) /
                  (2 * (u_dot_p0 - u_dot_pj));
//    if (cur_l > 0) {
//        cur_l = std::min(cur_l, has_precomputations ? max_length() : max_length(ref, u));
//    } else {
//        cur_l = std::max(cur_l, has_precomputations ? -max_length() : -max_length(ref, -u));
//    }
    if (cur_l > 0) {
        cur_l = std::min(cur_l, max_length(ref, u));
    } else {
        cur_l = std::max(cur_l, -max_length(ref, -u));
    }
    *length = cur_l;
}

void EuclideanKernel::intersect_ray_binsearch(const dvector &ref, const dvector &u, int j0, const IndexSet &ignore,
                                              int *best_j, ftype *best_l) const {
    if (j0 < 0) {
        throw std::runtime_error("Insertion cells are not yet supported for BIN_SEARCH strategy");
    }
    ftype max_l = max_length(ref, u);

    *best_j = -1;
    *best_l = max_l;

    // guaranteed to be smaller than MAX_RAY_LENGTH
    static ftype starting_length = std::min(max_length(), MAX_RAY_LENGTH);

    ftype up = u.dot(points.col(j0));
    ftype ds_i_j0 = (ref - points.col(j0)).squaredNorm();


    bool right_fixed = max_l < starting_length;
    ftype left_l = 0;
    ftype right_l = std::min(max_l, starting_length);

    int it = 0;
    dvector point;
    // 1 stage: find rightmost bound
    for (; it < MAX_BIN_SEARCH_ITER && !right_fixed; it++) {
        point = ref + u * right_l;
        ftype max_dist_sqr = (point - points.col(j0)).squaredNorm();
        ftype tmp = -1;
        int j = tree.find_nn(point, &tmp, max_dist_sqr, ignore);
        if (j < 0) {
            left_l = right_l;
            right_l = right_l * 2;
            if (right_l >= std::min(MAX_RAY_LENGTH, max_l)) {
                right_l = std::min(MAX_RAY_LENGTH, max_l);
                right_fixed = true;
            }
        } else {
            right_fixed = true;

            ftype ds_i_j = (ref - points.col(j)).squaredNorm();
            ftype cur_l = (ds_i_j - ds_i_j0) / (2 * (u.dot(points.col(j)) - up));
            //right_l = std::min(cur_l, mid_l);
            right_l = math::min(cur_l, right_l);
            *best_j = j;
        }
    }



    int left_counter = 0;
    int MAX_LEFT_COUNTER = 1;
    for (; it < MAX_BIN_SEARCH_ITER && right_l - left_l > BIN_SEARCH_MIN_INTERVAL; it++) {
        ftype mid_l = left_counter >= MAX_LEFT_COUNTER ? right_l :
                      BIN_SEARCH_SPLIT_PARAMETER * (right_l - left_l);
        point = ref + u * mid_l;
        ftype max_dist_sqr = (point - points.col(j0)).squaredNorm();
        ftype tmp = -1;
        int j = tree.find_nn(point, &tmp, max_dist_sqr,
                             left_counter >= MAX_LEFT_COUNTER ? ignore.append(*best_j) : ignore);
        if (j < 0) {
            left_l = std::max(left_l, mid_l);
            left_counter++;
        } else {
            assert(!ignore.contains(j));
            ftype ds_i_j = (ref - points.col(j)).squaredNorm();
            ftype cur_l = (ds_i_j - ds_i_j0) / (2 * (u.dot(points.col(j)) - up));
            //right_l = std::min(cur_l, mid_l);
            right_l = math::min(cur_l, right_l);
            *best_j = j;
            left_counter = 0;
        }
    }
    if (left_l - right_l > BIN_SEARCH_MIN_INTERVAL) {
        *best_j = -1;
    }
    if (*best_j >= 0) {
        ftype ds_i_j = (ref - points.col(*best_j)).squaredNorm();
        *best_l = (ds_i_j - ds_i_j0) / (2 * (u.dot(points.col(*best_j)) - up));
        #pragma omp critical
        starting_length = std::min(starting_length * (1 - STARTING_LENGTH_UPDATE_ALPHA)
                                   + *best_l * STARTING_LENGTH_UPDATE_ALPHA, MAX_RAY_LENGTH);
    } else {
        *best_l = right_l;
    }

    // final check
    if (*best_j >= 0 && !std::isfinite(*best_l)) {
        *best_j = -1;
    }

}

void SphericalKernel::intersect_ray_bruteforce(const dvector &ref, const dvector &u, int j0, const IndexSet &ignore,
                                               int *best_j_arr, ftype *best_l_arr) const {
    best_j_arr[0] = best_j_arr[1] = -1;
    best_l_arr[0] = max_length(ref, u);
    best_l_arr[1] = -max_length(ref, -u);

    int n_points = points.cols();

    auto a = points.col(j0);
    ftype pa = ref.dot(a);
    ftype ua = u.dot(a);

    // Go through all other points: O(n)
    for (int j = 0; j < n_points; j++) {
        if (ignore.contains(j)) {
            continue;
        }
        auto b = points.col(j);
        ftype pb = ref.dot(b);
        ftype ub = u.dot(b);
        ftype cur_l = atan((pa - pb) / (ub - ua));
        if (cur_l < 0) {
            cur_l += PI_ftype;
        }
        if (cur_l >= 0 && (best_j_arr[0] < 0 || cur_l < best_l_arr[0])) {
            best_j_arr[0] = j;
            best_l_arr[0] = cur_l;
        }
        cur_l -= PI_ftype;
        if (cur_l <= 0 && (best_j_arr[1] < 0 || cur_l > best_l_arr[1])) {
            // never used
            best_j_arr[1] = j;
            best_l_arr[1] = cur_l;
        }
    }
}

void SphericalKernel::intersect_ray_binsearch(const dvector &ref, const dvector &u, int j0, const IndexSet &ignore, int *best_j,
                                              ftype *best_l) const {
    *best_j = -1;
    *best_l = max_length(ref, u);

    ftype up = u.dot(points.col(j0));
    ftype ds_i_j0 = (ref - points.col(j0)).squaredNorm();

    ftype left_l = 0;
    ftype right_l = MAX_RAY_LENGTH;
    int left_counter = 0;
    int MAX_LEFT_COUNTER = 1;
    for (int it = 0; it < MAX_BIN_SEARCH_ITER && right_l - left_l > BIN_SEARCH_MIN_INTERVAL; it++) {
        ftype mid_l = left_counter >= MAX_LEFT_COUNTER ? right_l :
                      BIN_SEARCH_SPLIT_PARAMETER * (right_l - left_l);
        dvector point = cos(mid_l) * ref + sin(mid_l) * u;
        // euclidean metric is used to utilize the kd-tree!
        ftype max_dist_sqr = (point - points.col(j0)).squaredNorm();
        ftype tmp = -1;
        int j = tree.find_nn(point, &tmp, max_dist_sqr,
                             left_counter >= MAX_LEFT_COUNTER ? ignore.append(*best_j) : ignore);
        if (j < 0) {
            left_l = mid_l;
            left_counter++;
        } else {
            assert(!ignore.contains(j));
            ftype ds_i_j = (ref - points.col(j)).squaredNorm();
            ftype cur_l = (ds_i_j - ds_i_j0) / (2 * (u.dot(points.col(j)) - up));
            //right_l = std::min(cur_l, mid_l);
            right_l = cur_l;
            *best_j = j;
            left_counter = 0;
        }
    }
    if (*best_j >= 0) {
        ftype ds_i_j = (ref - points.col(*best_j)).squaredNorm();
        *best_l = (ds_i_j - ds_i_j0) / (2 * (u.dot(points.col(*best_j)) - up));
    }
}

int EuclideanKernel::nearest_point_extra(const dvector &ref, ftype within_radius,
                                         const IndexSet &ignore, ftype *distance) const {
    ftype tmp = -1;
    if (within_radius >= 0) {
        within_radius = sqr(within_radius);
    }
    int idx = tree.find_nn(ref, &tmp, within_radius, ignore);
    if (distance && tmp >= 0) {
        *distance = sqrt(tmp);
    }
    return idx;
}

int SphericalKernel::nearest_point_extra(const dvector &ref, ftype within_radius,
                                         const IndexSet &ignore, ftype *distance) const {
    ftype tmp = -1;
    if (within_radius >= 0) {
        within_radius = std::min(within_radius, PI_ftype);
        within_radius = sqr(2 * sin(within_radius / 2));
    }
    int idx = tree.find_nn(ref, &tmp, within_radius, ignore);
    if (distance && tmp >= 0) {
        *distance = 2 * asin(sqrt(tmp) / 2);
    }
    return idx;
}

ftype EuclideanKernel::distance(const dvector &a, const dvector &b) const {
    return (a - b).norm();
}

ftype SphericalKernel::distance(const dvector &a, const dvector &b) const {
    return acos(a.dot(b));
}

void EuclideanKernel::project_to_tangent_space_inplace(const dvector &at,
                                                       dvector &u) const {
    // nop
}

void SphericalKernel::project_to_tangent_space_inplace(const dvector &at,
                                                       dvector &u) const {
    u = u - u.dot(at) * at;
}

dvector EuclideanKernel::move_along_ray(const dvector &start,
                                        const dvector &direction,
                                        ftype length) const {
    return start + direction * length;
}

dvector SphericalKernel::move_along_ray(const dvector &start,
                                        const dvector &direction,
                                        ftype length) const {
    return start * cos(length) + direction * sin(length);
}

void SphericalKernel::update_inserted_points(int n_new_points) {
    throw std::runtime_error("not implemented");
}

ftype Unbounded::max_length(const const_dvector_ref &ref, const const_dvector_ref &u) const {
    return INF_ftype;
}

BoundingBox::BoundingBox(int dim, ftype half_width) :
        BoundingBox(dvector::Ones(dim) * (-half_width), dvector::Ones(dim) * half_width) { }

BoundingBox::BoundingBox(const dvector &mn, const dvector &mx) : mn(mn), mx(mx) {
    _max_length = (mx - mn).norm();
}

ftype BoundingBox::max_length(const const_dvector_ref &ref, const const_dvector_ref &u) const {
    Eigen::Matrix<ftype, dim_delta<DATA_DIM, DATA_DIM>(), 1> t(mn.size() + mx.size());
    Eigen::Matrix<ftype, dim_delta<DATA_DIM, DATA_DIM>(), 1> inf =
            Eigen::Matrix<ftype, dim_delta<DATA_DIM, DATA_DIM>(), 1>::Constant(mn.size() + mx.size(), INF_ftype);
    t << ((mn.array() - ref.array()) / u.array()), ((mx.array() - ref.array()) / u.array());
    t = (t.array() < 0).select(inf, t);
    return t.minCoeff();
}

BoundingSphere::BoundingSphere(int dim, ftype radius) :
        BoundingSphere(dvector::Zero(dim), radius) {}

BoundingSphere::BoundingSphere(const dvector &center, ftype radius) :
        center(center), radius(radius) { }

ftype BoundingSphere::max_length(const const_dvector_ref &ref, const const_dvector_ref &u) const {
    dvector diff = ref - center;
    ftype b = u.dot(diff);
    ftype c = diff.dot(diff) - radius * radius;
    ftype d = b * b - c;
    if (d < 0) {
        return 0; // already outside?
    }
    return -b + sqrt(d);
}

ftype Unbounded::max_length() const {
    return INF_ftype;
}

bool Unbounded::contains(const const_dvector_ref &ref) const {
    return true;
}

ftype BoundingBox::max_length() const {
    return _max_length;
}

bool BoundingBox::contains(const const_dvector_ref &ref) const {
    return (ref.array() > mn.array()).all() && (ref.array() < mx.array()).all();
}

const dvector &BoundingBox::get_mn() const {
    return mn;
}

const dvector &BoundingBox::get_mx() const {
    return mx;
}

ftype BoundingSphere::max_length() const {
    return radius * 2;
}

bool BoundingSphere::contains(const const_dvector_ref &ref) const {
    return (ref - center).squaredNorm() < radius * radius;
}

const dvector &BoundingSphere::get_center() const {
    return center;
}

ftype BoundingSphere::get_radius() const {
    return radius;
}

void EuclideanKernel::reset_ref_mat(int ref_n) {
    int data_n = points.cols();
    precomputed_ref_squared = dynvector::Constant(ref_n, NAN_ftype);
    precomputed_ref_dot_data = dynmatrix::Constant(ref_n, data_n, NAN_ftype);
}

void EuclideanKernel::reset_u_mat(int u_n) {
    int data_n = points.cols();
    precomputed_u_dot_data = dynmatrix::Constant(u_n, data_n, NAN_ftype);
}

void EuclideanKernel::precompute(const dmatrix &ref_mat, const dmatrix &u_mat) {
    reset_ref_mat(ref_mat.cols());
    reset_u_mat(u_mat.cols());

    int data_n = points.cols();
    int ref_n = ref_mat.cols();
    int u_n = u_mat.cols();

    std::cout << "Precomputing data squared" << std::endl;
    my_tqdm bar(data_n);
    #pragma omp parallel for
    for (int j = 0; j < data_n; j++) {
        bar.atomic_iteration();
        dvector data_j = points.col(j);
        precomputed_data_squared(j, 0) = data_j.squaredNorm();
    }
    bar.bar().finish();

    std::cout << "Precomputing ref squared" << std::endl;
    bar = my_tqdm(ref_n);
    #pragma omp parallel for
    for (int j = 0; j < ref_n; j++) {
        bar.atomic_iteration();
        dvector ref_j = ref_mat.col(j);
        precomputed_ref_squared(j, 0) = ref_j.squaredNorm();
    }
    bar.bar().finish();

    std::cout << "Precomputing ref dot data" << std::endl;
    bar = my_tqdm(data_n);
    #pragma omp parallel for
    for (int j = 0; j < data_n; j++) {
        bar.atomic_iteration();
        dvector data_j = points.col(j);
        dynvector col(ref_n, 1);
        for (int i = 0; i < ref_n; i++) {
            col(i, 0) = ref_mat.col(i).dot(data_j);
        }
        precomputed_ref_dot_data.col(j) = col;
    }
    bar.bar().finish();

    std::cout << "Precomputing u dot data" << std::endl;
    bar = my_tqdm(data_n);
    #pragma omp parallel for
    for (int j = 0; j < data_n; j++) {
        bar.atomic_iteration();
        dvector data_j = points.col(j);
        dynvector col(u_n, 1);
        for (int i = 0; i < u_n; i++) {
            col(i, 0) = u_mat.col(i).dot(data_j);
        }
        precomputed_u_dot_data.col(j) = col;
    }
    bar.bar().finish();
}

const ptr<Bounds> &EuclideanKernel::get_bounds() const {
    return bounds;
}

void EuclideanKernel::update_inserted_points(int n_new_points) {
    int n_old = precomputed_data_squared.rows();
    precomputed_data_squared.conservativeResize(n_old + n_new_points);
    precomputed_data_squared.block(n_old, 0, n_new_points, 1).setConstant(NAN_ftype);
    // todo: extend other "precomputed" matrices
    // ......

    tree.update_inserted_points();

}
