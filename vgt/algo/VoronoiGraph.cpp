#include "VoronoiGraph.h"

#include <memory>

#include "cnpy.h"

const VoronoiGraph::Polytope VoronoiGraph::NONE = VoronoiGraph::Polytope();

VoronoiGraph::Polytope::Polytope(const IndexSet &dual, const dvector &ref) :
        dual(dual), ref(ref) {}

VoronoiGraph::Polytope::Polytope() : dual({}) {}

bool VoronoiGraph::Polytope::is_none() const {
    return dual.empty();
}

VoronoiGraph::VoronoiGraph(RayStrategyType strategy) : strategy(strategy) {
}

bool VoronoiGraph::is_vertex(const VoronoiGraph::Polytope &p) const {
    return p.dual.dim() == data_dim;
}

void VoronoiGraph::read_points(const std::string &filename) {
    cnpy::NpyArray data_npy = cnpy::npy_load(filename);
    points = npy2matrix(data_npy);
    n_points = points.cols();
    data_dim = points.rows();

    std::cout << "Initializing KD-tree" << std::endl;
    tree = std::make_shared<KDTree>(points);
    tree->init();
}

VoronoiGraph::Polytope VoronoiGraph::retrieve_vertex_nearby(int point_idx, RandomEngine &re) const {
    return retrieve_vertex_nearby(points.col(point_idx), re, point_idx);
}

VoronoiGraph::Polytope VoronoiGraph::retrieve_vertex_nearby(const dvector &ref, RandomEngine &re,
                                                            int nearest_idx) const {
    dvector cur(ref);
    if (nearest_idx < 0) {
        ftype tmp = -1;
        nearest_idx = tree->find_nn(cur, &tmp);
    }
    IndexSet equidistants = {nearest_idx};
    vec<dvector> normals;
    bool ok = true;
    for (int cur_dim = 0; ok && cur_dim < data_dim; cur_dim++) {
        ok = false;
        for (int retries = 0; !ok && retries < MAX_RETRIES; retries++) {
            dvector u = re.rand_on_sphere(data_dim);
            for (const dvector &norm : normals) {
                u = u - u.dot(norm) * norm;
            }
            u.normalize();

            int best_j = -1;
            ftype best_l = 0;
            if (strategy == BRUTE_FORCE) {
                brute_force(cur, u, nearest_idx, equidistants, &best_j, &best_l, true);
            } else if (strategy == BIN_SEARCH) {
                bin_search(cur, u, nearest_idx, equidistants, &best_j, &best_l);
            }

            if (best_j < 0) {
                continue;
            }

            IndexSet new_eq = equidistants.append(best_j);
            dvector new_cur = cur + u * best_l;

            if (!validate(new_eq, new_cur)) {
                #pragma omp atomic
                validations_failed++;
                continue;
            } else {
                #pragma omp atomic
                validations_ok++;
            }

            equidistants = new_eq;
            cur = new_cur;

            dvector new_norm = points.col(best_j) - points.col(nearest_idx);
            for (const dvector &norm : normals) {
                new_norm = new_norm - new_norm.dot(norm) * norm;
            }
            new_norm.normalize();
            normals.push_back(new_norm);
            ok = true;
        }
    }
    if (ok) {
        return Polytope(equidistants, cur);
    } else {
        return NONE;
    }
}

/**
 * Returns argsort of negative eigenvalues.
 */
vec<int> test_lambda(const svector &lambda) {
    vec<int> res;
    for (int i = 0; i < lambda.size(); i++) {
        if (lambda[i] < 0) {
            res.push_back(i);
        }
    }
    std::sort(res.begin(), res.end(), [&](int a, int b) {
        return lambda[a] < lambda[b];
    });
    return res;
}

VoronoiGraph::Polytope VoronoiGraph::retrieve_vertex(const dvector &point, RandomEngine &re,
                                                     svector *coordinates) const {
    // Get an initial vertex nearby
    Polytope vertex = retrieve_vertex_nearby(point, re, -1);
    if (vertex.is_none()) { // potentially due to numeric issues
        return NONE;
    }

    svector q_vector(data_dim + 1);
    q_vector.head(data_dim) = point;
    q_vector[data_dim] = 1;

    svector lambda;
    for (int step = 0; step < VISIBILITY_WALK_MAX_STEPS; step++) {
        Eigen::Matrix<ftype, dim_delta<1>(), dim_delta<1>()> coords(data_dim + 1, data_dim + 1);
        for (int i = 0; i <= data_dim; i++) {
            coords.col(i).head(data_dim) = points.col(vertex.dual[i]);
            coords(data_dim, i) = 1;
        }
        lambda = coords.colPivHouseholderQr().solve(q_vector);
        vec<int> walk_directions = test_lambda(lambda);

        if (walk_directions.empty()) {
            break;
        }

        Polytope new_vertex;
        for (int i = 0; i < walk_directions.size() && new_vertex.is_none(); i++) {
            new_vertex = get_neighbor(vertex, walk_directions[i], re);
        }
        if (new_vertex.is_none()) {
            return NONE;
        }
        vertex = new_vertex;
    }

    if (coordinates) {
        *coordinates = lambda;
    }
    return vertex;
}

// todo re is not really needed, may be removed from the method definition later
VoronoiGraph::Polytope VoronoiGraph::get_neighbor(const VoronoiGraph::Polytope &vertex, int index,
                                                  RandomEngine &re) const {
    ensure(is_vertex(vertex), "`v` should be a Voronoi vertex");
    IndexSet edge = vertex.dual.remove_at(index);

    // Find the direction vector. May be done faster (right now - d^3)
    // find all normalizers
    vec<dvector> normals;
    for (int p = 1; p < data_dim; p++) {
        dvector v = points.col(edge[p]) - points.col(edge[0]);
        for (const dvector &norm : normals) {
            v = v - v.dot(norm) * norm;
        }
        v.normalize();
        normals.push_back(v);
    }
    // u -- direction vector
    dvector u = re.rand_on_sphere(data_dim);
    for (const dvector &norm : normals) {
        u = u - u.dot(norm) * norm;
    }
    u.normalize();
    // determine the correct direction of u
    if (u.dot(points.col(vertex.dual[index]) - points.col(edge[0])) > 0) {
        u = -u;
    }

    // find the other end of the edge
    int best_j = -1;
    ftype best_l = -1;

    if (strategy == BRUTE_FORCE) {
        brute_force(vertex.ref, u, edge[0], vertex.dual, &best_j, &best_l, false);
    } else {
        assert(strategy == BIN_SEARCH);
        bin_search(vertex.ref, u, edge[0], vertex.dual, &best_j, &best_l);
    }

    if (best_j >= 0) {
        // move to the next point (otherwise, stay)
        IndexSet next_vertex = edge.append(best_j);
        dvector next_ref = vertex.ref + u * best_l;
        if (validate(next_vertex, next_ref)) {
            #pragma omp atomic
            validations_ok++;
            return Polytope(next_vertex, next_ref);
        } else {
            #pragma omp atomic
            validations_failed++;
            return NONE;
        }
    }

    return NONE;
}

vec<VoronoiGraph::Polytope> VoronoiGraph::get_neighbors(const VoronoiGraph::Polytope &vertex,
                                                        RandomEngine &re) const {
    vec<Polytope> result;
    for (int i = 0; i <= data_dim; i++) {
        result.push_back(get_neighbor(vertex, i, re));
    }
    return result;
}

VoronoiGraph::Polytope
VoronoiGraph::sample_ray(const VoronoiGraph::Polytope &p, RandomEngine &re,
                         ptr<const vec<dvector>> orthogonal_complement, bool bidirectional) const {
    ensure(!p.is_none(), "p should not be NONE");
    if (!orthogonal_complement) {
        vec<dvector> normals;
        for (int i = 1; i < data_dim; i++) {
            dvector v = points.col(p.dual[i]) - points.col(p.dual[0]);
            for (const dvector &norm : normals) {
                v = v - v.dot(norm) * norm;
            }
            v.normalize();
            normals.push_back(v);
        }
        orthogonal_complement = std::make_shared<const vec<dvector>>(normals);
    }
    dvector u = re.rand_on_sphere(data_dim);
    for (const dvector &norm : *orthogonal_complement) {
        u = u - u.dot(norm) * norm;
    }
    u.normalize();

    int best_j = -1;
    ftype best_l = 0;
    if (strategy == BRUTE_FORCE) {
        brute_force(p.ref, u, p.dual[0], p.dual, &best_j, &best_l, bidirectional);
    } else if (strategy == BIN_SEARCH) {
        bin_search(p.ref, u, p.dual[0], p.dual, &best_j, &best_l);
    }

    if (best_j < 0) {
        return NONE;
    }

    IndexSet new_eq = p.dual.append(best_j);
    dvector new_ref = p.ref + u * best_l;

    if (!validate(new_eq, new_ref)) {
        #pragma omp atomic
        validations_failed++;
        return NONE;
    } else {
        #pragma omp atomic
        validations_ok++;
        return Polytope(new_eq, new_ref);
    }
}

RayStrategyType VoronoiGraph::get_strategy() const {
    return strategy;
}

void VoronoiGraph::set_strategy(RayStrategyType new_strategy) {
    VoronoiGraph::strategy = new_strategy;
}

void VoronoiGraph::brute_force(const dvector &ref, const dvector &u, int j0, const IndexSet &ignore,
                               int *best_j, ftype *best_l, bool bidirectional) const {
    int tmp_j[2] = {-1, -1};
    ftype tmp_l[2] = {0, 0};

    ftype up = u.dot(points.col(j0));
    ftype ds_i_j0 = (ref - points.col(j0)).squaredNorm();
    // Go through all other points: O(n)
    for (int j = 0; j < n_points; j++) {
        if (ignore.contains(j)) {
            continue;
        }
        ftype ds_i_j = (ref - points.col(j)).squaredNorm();
        ftype cur_l = (ds_i_j - ds_i_j0) / (2 * (u.dot(points.col(j)) - up));
        if (cur_l >= 0 && (tmp_j[0] < 0 || cur_l < tmp_l[0])) {
            tmp_j[0] = j;
            tmp_l[0] = cur_l;
        }
        if (bidirectional && cur_l <= 0 && (tmp_j[1] < 0 || cur_l > tmp_l[1])) {
            tmp_j[1] = j;
            tmp_l[1] = cur_l;
        }
    }
    int picked = 0;
    if (bidirectional && tmp_j[0] < 0) {
        picked = 1;
    }
    *best_j = tmp_j[picked];
    *best_l = tmp_l[picked];
}

void VoronoiGraph::bin_search(const dvector &ref, const dvector &u, int j0, const IndexSet &ignore,
                              int *best_j, ftype *best_l) const {
    ftype up = u.dot(points.col(j0));
    ftype ds_i_j0 = (ref - points.col(j0)).squaredNorm();

    ftype left_l = 0;
    ftype right_l = 100;       // todo change relatively to the diameter of data
    int left_counter = 0;
    int MAX_LEFT_COUNTER = 1;
    for (int it = 0; it < BIN_SEARCH_NITER && right_l - left_l > BIN_SEARCH_EPS; it++) {
        ftype mid_l = left_counter >= MAX_LEFT_COUNTER ? right_l :
                      BIN_SEARCH_PARAMETER * (right_l - left_l);
        dvector point = ref + u * mid_l;
        ftype max_dist_sqr = (point - points.col(j0)).squaredNorm();
        ftype tmp = -1;
        int j = tree->find_nn(point, &tmp, max_dist_sqr,
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

bool VoronoiGraph::validate(const vec<int> &is, const dvector &ref) const {
    ftype min_dist = (ref - points.col(is[0])).squaredNorm();
    ftype max_dist = min_dist;
    for (size_t i = 1; i < is.size(); i++) {
        ftype dst = (ref - points.col(is[i])).squaredNorm();
        min_dist = std::min(min_dist, dst);
        max_dist = std::max(max_dist, dst);
    }
    if (max_dist - min_dist > VALIDATION_EPS) {
        return false;
    }
    ftype other_dist = -1;
    return tree->find_nn(ref, &other_dist, max_dist - VALIDATION_EPS, is) < 0;
}

dmatrix VoronoiGraph::get_data() const {
    return points;
}

int VoronoiGraph::get_data_size() const {
    return n_points;
}

int VoronoiGraph::get_data_dim() const {
    return data_dim;
}

void VoronoiGraph::reset_failure_rate_counters() const {
    validations_ok = 0;
    validations_failed = 0;
}

std::pair<long long, long long> VoronoiGraph::get_failure_rate() const {
    return std::make_pair(validations_ok, validations_failed);
}

void VoronoiGraph::print_validations_info() const {
    printf("Validations failed: %lld out of %lld (%.2f%%)\n",
           validations_failed, validations_ok + validations_failed,
           100 * float(validations_failed)/float(validations_ok + validations_failed));
    fflush(stdout);
}
