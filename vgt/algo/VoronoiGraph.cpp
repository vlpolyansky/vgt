#include "VoronoiGraph.h"

#include <memory>

#include "cnpy.h"
#include "kernels_gpu.h"

const VoronoiGraph::Polytope VoronoiGraph::NONE = VoronoiGraph::Polytope();

VoronoiGraph::Polytope::Polytope(const IndexSet &dual, const dvector &ref) :
        dual(dual), ref(ref) {}

VoronoiGraph::Polytope::Polytope() : dual({}) {}

bool VoronoiGraph::Polytope::is_none() const {
    return dual.empty();
}

VoronoiGraph::VoronoiGraph(RayStrategyType strategy, DataType data_type) :
        strategy(strategy), data_type(data_type) {
}

Kernel& VoronoiGraph::get_kernel() const {
    return *kernel;
}

bool VoronoiGraph::is_vertex(const VoronoiGraph::Polytope &p) const {
    return p.dual.dim() == data_dim;
}

void VoronoiGraph::initialize(const dmatrix &data, ptr<Bounds> bounds) {
    points = data;
    n_points = points.cols();
    ambient_dim = points.rows();

    // Initialize kernel
    switch (data_type) {
        case EUCLIDEAN:
            if (!bounds) {
                bounds = std::make_shared<Unbounded>();
            }
            if (strategy == BRUTE_FORCE_GPU) {
                kernel = std::make_shared<EuclideanKernelGPU>(points, bounds);
            } else {
                kernel = std::make_shared<EuclideanKernel>(points, bounds);
            }
            data_dim = ambient_dim;
            break;
        case SPHERICAL:
            ensure(!bounds, "Data bounds are not allowed for the spherical data");
            kernel = std::make_shared<SphericalKernel>(points);
            data_dim = ambient_dim - 1;
            break;
        case TOROIDAL:
        default:
            throw std::runtime_error("Current data type is not supported");
    }
}

void VoronoiGraph::read_points(const std::string &filename, ptr<Bounds> bounds) {
    initialize(npy2matrix(cnpy::npy_load(filename)), bounds);
}

int VoronoiGraph::get_containing_voronoi_cell(const dvector &ref) const {
    return get_kernel().nearest_point(ref);
}

VoronoiGraph::Polytope VoronoiGraph::retrieve_vertex_nearby(int point_idx, RandomEngine &re) const {
    return retrieve_vertex_nearby(points.col(point_idx), re, point_idx);
}

VoronoiGraph::Polytope VoronoiGraph::retrieve_vertex_nearby(const dvector &ref, RandomEngine &re,
                                                            int nearest_idx) const {
    dmatrix ref_mat(ambient_dim, 1);
    ref_mat = ref;
    RandomEngineMultithread rem(re);
    vec<int> nearest_idx_vec;
    if (nearest_idx >= 0) {
        nearest_idx_vec = {nearest_idx};
    }
    return retrieve_vertices_nearby(ref_mat, rem, nearest_idx_vec)[0];
}

vec<VoronoiGraph::Polytope>
VoronoiGraph::retrieve_vertices_nearby(const dmatrix &ref_mat, RandomEngineMultithread &re, vec<int> nearest_idx_vec) const {
    #pragma omp parallel
    re.fix_random_engines();

    int n_queries = ref_mat.cols();

    // `cur` is the current reference point, it changes after every ray projection
    dmatrix cur_mat(ref_mat);
    // determine the initial Voronoi cell

    if (nearest_idx_vec.empty()) {
        nearest_idx_vec = vec<int>(n_queries);
        #pragma omp parallel for
        for (int i = 0; i < n_queries; i++) {
            nearest_idx_vec[i] = get_containing_voronoi_cell(cur_mat.col(i));
        }
    }
    // `dual` contains vertices of a delaunay simplex, dual of which contains `cur`
    vec<IndexSet> dual_vec(n_queries, vec<int>());
    for (int i = 0; i < n_queries; i++) {
        dual_vec[i] = {nearest_idx_vec[i]};
    }
    // `normals` describes the basis of the orthogonal complement of the current polytope
    vec<vec<dvector>> normals_vec(n_queries);
    for (int cur_dim = 0; cur_dim < data_dim; cur_dim++) {
        // try to find a finite direction within the polytope
        // should happen with probability 1, but just in case -- retried a few times
        dmatrix u_mat(ambient_dim, n_queries);
        #pragma omp parallel
        for (int i = 0; i < n_queries; i++) {
            if (dual_vec[i].empty()) continue;
            // random direction `u` is picked uniformly ...
            dvector u = re.current().rand_on_sphere(ambient_dim);
            // ..., projected onto the polytope...
            for (const dvector &norm : normals_vec[i]) {
                u = u - u.dot(norm) * norm;
            }
            // ... including the projection onto the tangent hyperplane of the data manifold
            // (needed for spherical data) ...
            get_kernel().project_to_tangent_space_inplace(cur_mat.col(i), u);
            // ..., and normalized
            u.normalize();
            u_mat.col(i) = u;
        }

        vec<int> best_j_vec(n_queries, -1);
        vec<ftype> best_l_vec(n_queries, 0);
        if (get_strategy() != BRUTE_FORCE_GPU) {
            #pragma omp parallel for
            for (int i = 0; i < n_queries; i++) {
                if (dual_vec[i].empty()) {
                    continue;
                }
                get_kernel().intersect_ray(strategy, cur_mat.col(i),
                                           u_mat.col(i), nearest_idx_vec[i], dual_vec[i],
                                           &best_j_vec[i], &best_l_vec[i],
                                           Kernel::ANY_INTERSECTION);
            }

        } else {
            EuclideanKernelGPU &kernel_gpu = dynamic_cast<EuclideanKernelGPU &>(get_kernel());
            vec<int> best_j_pos(n_queries, -1);
            vec<int> best_j_neg(n_queries, -1);
            vec<int> u_indices(n_queries);
            for (int i = 0; i < n_queries; i++) {
                u_indices[i] = i;
            }
            if (cur_dim == -1) {
                kernel_gpu.reset_reference_points_gpu(cur_mat, nearest_idx_vec);
            } else {
                kernel_gpu.reset_reference_points_gpu(cur_mat, dual_vec);
            }
            kernel_gpu.reset_rays_gpu(u_mat);
            kernel_gpu.intersect_ray_indexed_gpu(u_indices, &best_j_pos, &best_j_neg);

            #pragma omp parallel for
            for (int i = 0; i < n_queries; i++) {
                best_j_vec[i] = best_j_pos[i] >= 0 ? best_j_pos[i] : best_j_neg[i];
                if (best_j_vec[i] >= 0) {
                    kernel_gpu.intersect_ray_precomputed(
                            cur_mat.col(i), u_mat.col(i), nearest_idx_vec[i],
                            best_j_vec[i], &best_l_vec[i]);
                }
            }
        }

        #pragma omp parallel for
        for (int i = 0; i < n_queries; i++) {
            if (best_j_vec[i] < 0) {
                // todo retry with a different direction
                cur_mat.col(i).setConstant(NAN_ftype);
                dual_vec[i] = {};
                continue;
            }
            // the dual of the next boundary
            IndexSet new_dual = dual_vec[i].append(best_j_vec[i]);
            // a new reference point on that boundary
            dvector new_cur = get_kernel().move_along_ray(cur_mat.col(i), u_mat.col(i), best_l_vec[i]);

            // check that the new boundary is valid
            if (!validate(new_dual, new_cur)) {
                #pragma omp atomic
                validations_failed++;
                dual_vec[i] = {};
                continue;
            } else {
                #pragma omp atomic
                validations_ok++;
            }

            // update the current polytope
            dual_vec[i] = new_dual;
            cur_mat.col(i) = new_cur;

            // add a new vector to the basis of the complement
            dvector new_norm = points.col(best_j_vec[i]) - points.col(nearest_idx_vec[i]);
            for (const dvector &norm : normals_vec[i]) {
                new_norm = new_norm - new_norm.dot(norm) * norm;
            }
            new_norm.normalize();
            normals_vec[i].push_back(new_norm);
        }
    }

    vec<Polytope> result(n_queries);
    for (int i = 0; i < n_queries; i++) {
        if (!dual_vec[i].empty()) {
            result[i] = Polytope(dual_vec[i], cur_mat.col(i));
        }
    }
    return result;
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
    dmatrix queries(ambient_dim, 1);
    queries = point;
    RandomEngineMultithread rem(re);
    vec<svector> all_coordinates;
    vec<Polytope> vertices = retrieve_vertices(queries, rem, coordinates ? &all_coordinates : nullptr);

    if (coordinates) {
        *coordinates = all_coordinates[0];
    }
    return vertices[0];
}

vec<VoronoiGraph::Polytope>
VoronoiGraph::retrieve_vertices(const dmatrix &queries, RandomEngineMultithread &re, vec<svector> *coordinates) const {
    int n_queries = queries.cols();
    vec<VoronoiGraph::Polytope> result(n_queries);
    #pragma omp parallel
    re.fix_random_engines();

    // Get an initial vertex nearby
    vec<Polytope> vertices = retrieve_vertices_nearby(queries, re, {});
    vec<int> done(n_queries);
    vec<set_t<IndexSet>> was(n_queries);
    vec<svector> lambdas(n_queries);

    for (int step = 0; step < VISIBILITY_WALK_MAX_STEPS; step++) {
        vec<int> unfinished_ids;
        for (int i = 0; i < n_queries; i++) {
            if (!done[i] && !vertices[i].is_none()) {
                unfinished_ids.push_back(i);
            }
        }

        if (unfinished_ids.empty()) {
            break;
        }

        vec<vec<int>> walk_directions(n_queries);

        #pragma omp parallel for
        for (int ii = 0; ii < unfinished_ids.size(); ii++) {
            int i = unfinished_ids[ii];

            Polytope &vertex = vertices[i];

            svector q_vector(ambient_dim + 1);
            q_vector.head(ambient_dim) = queries.col(i);
            q_vector[ambient_dim] = 1;

            was[i].insert(vertex.dual);

            Eigen::Matrix<ftype, dim_delta<DATA_DIM, 1>(), dim_delta<DATA_DIM, 1>()>
                    coords(ambient_dim + 1, ambient_dim + 1);
            for (int j = 0; j <= ambient_dim; j++) {
                coords.col(j).head(ambient_dim) = points.col(vertex.dual[j]);
                coords(ambient_dim, j) = 1;
            }
            lambdas[i] = coords.colPivHouseholderQr().solve(q_vector);
            walk_directions[i] = test_lambda(lambdas[i]);

            if (walk_directions[i].empty()) {
                done[i] = 1;
                continue;
            }
        }

        vec<int> to_move_ids;
        for (int i : unfinished_ids) {
            if (!done[i]) {
                to_move_ids.push_back(i);
            }
        }

        for (int j = 0; !to_move_ids.empty(); j++) {
            vec<int> cur_to_move_ids;
            vec<Polytope> cur_vertices;
            vec<int> cur_directions;

            for (int i : to_move_ids) {
                if (j >= walk_directions[i].size()) {
                    vertices[i] = NONE;
                    done[i] = true;
                    continue;
                }
                cur_to_move_ids.push_back(i);
                cur_vertices.push_back(vertices[i]);
                cur_directions.push_back(walk_directions[i][j]);
            }

            if (cur_to_move_ids.empty()) {
                break;
            }
            vec<Polytope> cur_neighbors = get_neighbors(cur_vertices, cur_directions, re);

            vec<int> next_to_move_ids;
            for (int ii = 0; ii < cur_to_move_ids.size(); ii++) {
                if (cur_neighbors[ii].is_none()) {
                    next_to_move_ids.push_back(cur_to_move_ids[ii]);
                } else {
                    vertices[cur_to_move_ids[ii]] = cur_neighbors[ii];
                }
            }

            to_move_ids = next_to_move_ids;
        }

        #pragma omp parallel for
        for (int ii = 0; ii < unfinished_ids.size(); ii++) {
            int i = unfinished_ids[ii];
            if (!vertices[i].is_none()) {
                if (was[i].find(vertices[i].dual) == was[i].end()) {
                    was[i].insert(vertices[i].dual);
                } else {
                    done[i] = true;
                    continue;
                }
            }
        }
    }

    if (coordinates) {
        *coordinates = lambdas;
    }
    return vertices;
}

vec<vec<VoronoiGraph::Polytope>>
VoronoiGraph::get_neighbors(const vec<VoronoiGraph::Polytope> &vertices, RandomEngineMultithread &re) const {
    int n_vertices = vertices.size();
    vec<vec<VoronoiGraph::Polytope>> result(n_vertices);
    if (get_strategy() != BRUTE_FORCE_GPU) {
        #pragma omp parallel
        {
            re.fix_random_engines();
            #pragma omp for schedule(dynamic)
            for (int i = 0; i < n_vertices; i++) {
                result[i] = get_neighbors(vertices[i], re.current());
            }
        }
    } else {
        EuclideanKernelGPU &kernel_gpu = dynamic_cast<EuclideanKernelGPU &>(get_kernel());


        throw std::runtime_error("not implemented yet");
    }
    return result;
}

vec<VoronoiGraph::Polytope>
VoronoiGraph::get_neighbors(const vec<Polytope> &vertices, const vec<int> &indices,
                            RandomEngineMultithread &re) const {
    #pragma omp parallel
    re.fix_random_engines();

    int n_vertices = vertices.size();
    vec<VoronoiGraph::Polytope> result(n_vertices);

    if (get_strategy() != BRUTE_FORCE_GPU) {
        #pragma omp parallel for
        for (int i = 0; i < n_vertices; i++) {
            result[i] = get_neighbor(vertices[i], indices[i], re.current());
        }
    } else {
        EuclideanKernelGPU &kernel_gpu = dynamic_cast<EuclideanKernelGPU &>(get_kernel());
        dmatrix ref_mat(ambient_dim, n_vertices);
        dmatrix u_mat(ambient_dim, n_vertices);
        vec<IndexSet> duals(n_vertices);
        vec<int> u_indices(n_vertices);

        #pragma omp parallel for
        for (int i = 0; i < n_vertices; i++) {
            ref_mat.col(i) = vertices[i].ref;
            duals[i] = vertices[i].dual;
            u_indices[i] = i;

            IndexSet edge = vertices[i].dual.remove_at(indices[i]);
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
            dvector u = re.current().rand_on_sphere(ambient_dim);
            for (const dvector &norm : normals) {
                u = u - u.dot(norm) * norm;
            }
            get_kernel().project_to_tangent_space_inplace(vertices[i].ref, u);
            u.normalize();
            // determine the correct direction of u
            if (u.dot(points.col(vertices[i].dual[indices[i]]) - points.col(edge[0])) > 0) {
                u = -u;
            }
            u_mat.col(i) = u;
        }

        // find the other end of the edge
        vec<int> best_j_pos(n_vertices, -1);
        vec<int> best_j_neg(n_vertices, -1);

        kernel_gpu.reset_reference_points_gpu(ref_mat, duals);
        kernel_gpu.reset_rays_gpu(u_mat);
        kernel_gpu.intersect_ray_indexed_gpu(u_indices, &best_j_pos, &best_j_neg);

        #pragma omp parallel for
        for (int i = 0; i < n_vertices; i++) {
            if (best_j_pos[i] >= 0) {
                ftype best_l;
                kernel_gpu.intersect_ray_precomputed(ref_mat.col(i), u_mat.col(i),
                                                     duals[i][0], best_j_pos[i], &best_l);
                // move to the next point (otherwise, stay)
                IndexSet next_vertex = vertices[i].dual.remove_at(indices[i]).append(best_j_pos[i]);
                dvector next_ref = kernel_gpu.move_along_ray(
                        ref_mat.col(i), u_mat.col(i), best_l);
                if (validate(next_vertex, next_ref)) {
                    #pragma omp atomic
                    validations_ok++;
                    result[i] = Polytope(next_vertex, next_ref);
                } else {
                    #pragma omp atomic
                    validations_failed++;
                }
            }
        }
    }
    return result;
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
    dvector u = re.rand_on_sphere(ambient_dim);
    for (const dvector &norm : normals) {
        u = u - u.dot(norm) * norm;
    }
    get_kernel().project_to_tangent_space_inplace(vertex.ref, u);
    u.normalize();
    // determine the correct direction of u
    if (u.dot(points.col(vertex.dual[index]) - points.col(edge[0])) > 0) {
        u = -u;
    }

    // find the other end of the edge
    int best_j = -1;
    ftype best_l = -1;

    get_kernel().intersect_ray(strategy, vertex.ref, u, edge[0], vertex.dual,
                               &best_j, &best_l, Kernel::RAY_INTERSECTION);

    if (best_j >= 0) {
        // move to the next point (otherwise, stay)
        IndexSet next_vertex = edge.append(best_j);
        dvector next_ref = get_kernel().move_along_ray(vertex.ref, u, best_l);
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
VoronoiGraph::cast_ray(const VoronoiGraph::Polytope &p, const dvector &direction,
                       ptr<const vec<dvector>> orthogonal_complement, bool any_direction,
                       ftype *length) const {
    ensure(!p.is_none(), "p should not be NONE");
    if (!orthogonal_complement) {
        vec<dvector> normals;
        for (int i = 1; i < p.dual.dim(); i++) {
            dvector v = points.col(p.dual[i]) - points.col(p.dual[0]);
            for (const dvector &norm : normals) {
                v = v - v.dot(norm) * norm;
            }
            v.normalize();
            normals.push_back(v);
        }
        orthogonal_complement = std::make_shared<const vec<dvector>>(normals);
    }
    dvector u = direction;
    for (const dvector &norm : *orthogonal_complement) {
        u = u - u.dot(norm) * norm;
    }
    get_kernel().project_to_tangent_space_inplace(p.ref, u);
    u.normalize();

    int best_j = -1;
    ftype best_l = 0;
    get_kernel().intersect_ray(strategy, p.ref, u, p.dual[0], p.dual,
                               &best_j, &best_l,
                               any_direction ? Kernel::ANY_INTERSECTION : Kernel::RAY_INTERSECTION);

    if (length) {
        *length = best_l;
    }

    if (best_j < 0) {
        return NONE;
    }

    IndexSet new_dual = p.dual.append(best_j);
    dvector new_ref = get_kernel().move_along_ray(p.ref, u, best_l);

    if (!validate(new_dual, new_ref)) {
        #pragma omp atomic
        validations_failed++;
        return NONE;
    } else {
        #pragma omp atomic
        validations_ok++;
        return Polytope(new_dual, new_ref);
    }
}

bool VoronoiGraph::validate(const vec<int> &is, const dvector &ref) const {
    Kernel &k = get_kernel();
    ftype min_dist = k.distance(ref, points.col(is[0]));
    ftype max_dist = min_dist;
    for (size_t i = 1; i < is.size(); i++) {
        ftype dst = k.distance(ref, points.col(is[i]));
        min_dist = std::min(min_dist, dst);
        max_dist = std::max(max_dist, dst);
    }
//    std::cout << min_dist << " " << max_dist << std::endl;
    if (max_dist - min_dist > VALIDATION_EPS) {
        return false;
    }
    return get_kernel().nearest_point_extra(ref, max_dist - VALIDATION_EPS, is) < 0;
}

const dmatrix& VoronoiGraph::get_data() const {
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

RayStrategyType VoronoiGraph::get_strategy() const {
    return strategy;
}

DataType VoronoiGraph::get_data_type() const {
    return data_type;
}

void VoronoiGraph::insert(const dmatrix &new_points) {
    int old_n = points.cols();
    int new_n = new_points.cols();
    n_points = old_n + new_n;
    points.conservativeResize(ambient_dim, n_points);
    points.block(0, old_n, ambient_dim, new_n) = new_points;

    get_kernel().update_inserted_points(new_n);
}
