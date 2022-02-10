
#include <cnpy.h>
#include <argparse.hpp>

#include "../utils.h"
#include "../algo/VoronoiGraph.h"
#include "../algo/kernels_gpu.h"

argparse::ArgumentParser prepare_parser() {
    argparse::ArgumentParser program("./significant_delaunay");
    program.add_description("Delaunay graph approximator based or first-order ray-sampling");

    program.add_argument("data.npy")
            .help("npy NxD data matrix of 32-bit floats");

    program.add_argument("--queries")
            .help("npy MxD query matrix of 32-bit floats")
            .default_value(std::string(""));

    program.add_argument("--out")
            .help("output filename for the graph with lines in form 'v_i v_j spherical'")
            .default_value(std::string("graph.npy"));

    program.add_argument("--out_dist")
            .help("output filename for "
                  "an npz data file containing 'indices', 'simplices', 'coefficients' and"
                  " (if 'values' are provided) 'estimates' matrices")
            .default_value(std::string(""));

    program.add_argument("--nrays")
            .help("number of rays from each point")
            .default_value(1000)
            .action(&parse_int);

    program.add_argument("--seed")
            .help("random seed")
            .default_value(239)
            .action(&parse_int);

    program.add_argument("--njobs")
            .help("number of parallel cpu threads (requires OpenMP)")
            .default_value(-1)
            .action(&parse_int);

    return program;
}

void save_graph(vec<std::map<int, float>> edges, const dmatrix &target, const dmatrix &source, bool bipartite,
                const std::string output_filename, const std::string distances_filename) {
    std::cout << "Saving the results" << std::endl;

    int target_n = target.cols();
    int source_n = source.cols();

    if (!bipartite) {
        // this is a quick fix for edges
        for (int from = 0; from < edges.size(); from++) {
            for (const auto &e: edges[from]) {
                int to = e.first;
                if (from > to && edges[to].find(from) == edges[to].end()) {
                    edges[to][from] = 0;
                }
            }
        }
    }

    int prefix = bipartite ? target_n : 0;
    size_t width = 3;
    std::string mode = "w";

    my_tqdm bar(source_n);

    #pragma omp parallel for ordered schedule(static,1)
    for (int from = 0; from < source_n; from++) {
        bar.atomic_iteration();
        vec<int> output;  // stores from, to, significance
        vec<float> distances;
        for (const auto &e: edges[from]) {
            int to = e.first;
            int significance = static_cast<int>(e.second); // careful, we expect weight as integer!
            if (from < to || bipartite) {
                if (!bipartite) {
                    auto other = edges[to].find(from);
                    if (other != edges[to].end()) {
                        significance += other->second;
                    }
                }
                ensure(significance > 0, "coverage is expected to be greater than zero");
                output.push_back(from + prefix);
                output.push_back(to);
                output.push_back(significance);
                if (!distances_filename.empty()) {
                    distances.push_back(static_cast<float>((source.col(from) - target.col(to)).norm()));
                }
            }
        }

        #pragma omp ordered
        {
            cnpy::npy_save(output_filename, &output[0], {output.size() / width, width}, mode);

            if (!distances_filename.empty()) {
                cnpy::npy_save(distances_filename, &distances[0], {distances.size(), 1}, mode);
            }
            mode = "a";
        }
    }
    bar.bar().finish();
}

int main(int argc, char **argv) {
    // TO BE REFACTORED
    argparse::ArgumentParser program = prepare_parser();
    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cout << err.what() << std::endl;
        std::cout << program;
        exit(0);
    }

    int njobs = program.get<int>("--njobs");
    if (njobs > 0) {
        omp_set_num_threads(njobs);
    }

    std::string data_filename = program.get("data.npy");
    std::string query_filename = program.get("--queries");
    bool bipartite = !query_filename.empty();

    int seed = program.get<int>("--seed");

    std::string output_filename = program.get("--out");
    std::string distances_filename = program.get("--out_dist");

    RandomEngineMultithread re(seed);

    dmatrix points = npy2matrix(cnpy::npy_load(data_filename));
    int data_n = points.cols();
    int dim = points.rows();

    dmatrix queries = !bipartite ? points : npy2matrix(cnpy::npy_load(query_filename));
    int queries_n = queries.cols();

    std::vector<std::map<int, float>> edges(queries_n);

    EuclideanKernelGPU kernel(points);

    int u_n = program.get<int>("--nrays");

    int max_block_size = -1;
    max_block_size = kernel.estimate_max_block_size();
    if (u_n < max_block_size) {
        max_block_size = kernel.estimate_max_block_size(u_n);
    }
    std::cout << "Maximum block size estimated as " << max_block_size << std::endl;

    int u_block_n = max_block_size < 0 ? 1 : (u_n + max_block_size - 1) / max_block_size;
    int ref_block_n = max_block_size < 0 ? 1 : (queries_n + max_block_size - 1) / max_block_size;
    int block_size = max_block_size < 0 ? math::max(queries_n, u_n) : max_block_size;

    for (int u_block_i = 0; u_block_i < u_block_n; u_block_i++) {
        int u_block_start = u_block_i * block_size;
        int u_block_end = math::min(u_block_start + block_size, u_n);
        // generate directions
        dmatrix u_mat = dmatrix(dim, u_block_end - u_block_start);
        for (int i = 0; i < u_mat.cols(); i++) {
            u_mat.col(i) = re.current().rand_on_sphere(dim);
        }

        for (int ref_block_i = 0; ref_block_i < ref_block_n; ref_block_i++) {
            std::cout << " * Block (" << u_block_i + 1 << ", " << ref_block_i + 1
                      << ") out of (" << u_block_n << ", " << ref_block_n << ") ..." << std::endl;
            int ref_block_start = ref_block_i * block_size;
            int ref_block_end = math::min(ref_block_start + block_size, queries_n);
//            dmatrix ref_mat = points(Eigen::all, Eigen::seq(ref_block_start, ref_block_end - 1));
            dmatrix ref_mat = queries.block(0, ref_block_start, queries.rows(), ref_block_end - ref_block_start);

            // precalculations
            kernel.reset_ref_mat(ref_mat.cols());
            kernel.reset_u_mat(u_mat.cols());
            std::cout << "Running GPU precalculations" << std::endl;
            vec<int> j0_list(ref_mat.cols());
            for (int i = 0; i < j0_list.size(); i++) {
                j0_list[i] = !bipartite ? ref_block_start + i : -1;
            }
            kernel.reset_reference_points_gpu(ref_mat, j0_list);
            kernel.reset_rays_gpu(u_mat);

            vec<int> best_j_pos((ref_block_end - ref_block_start) * (u_block_end - u_block_start));
            vec<int> best_j_neg((ref_block_end - ref_block_start) * (u_block_end - u_block_start));

            std::cout << "Running raycasting on GPU" << std::endl;
            kernel.intersect_ray_bruteforce_gpu(&best_j_pos, &best_j_neg);

            std::cout << "Aggregating results" << std::endl;
            my_tqdm bar(ref_block_end - ref_block_start);

            #pragma omp parallel
            {
                #pragma omp master
                {
                    if (u_block_i == 0 && ref_block_i == 0) {
                        std::cout << "Using " << omp_get_num_threads() << " threads" << std::endl;
                    }
                }
                re.fix_random_engines();

                #pragma omp for
                for (int ref_i = ref_block_start; ref_i < ref_block_end; ref_i++) {
                    bar.atomic_iteration();
                    std::map<int, float> &edges_from_ref = edges[ref_i];

                    for (int j = 0; j < u_mat.cols(); j++) {
                        int to = best_j_pos[(ref_i - ref_block_start) * u_mat.cols() + j];
                        if (to >= 0) {
                            auto e = edges_from_ref.find(to);
                            if (e != edges_from_ref.end()) {
                                e->second += 1.0f;
                            } else {
                                edges_from_ref[to] = 1.0f;
                            }
                        }
                    }
                }
            }
            bar.bar().finish();
            // ## end of ref loop ##
        }

        // ## end of u loop ##
    }

    save_graph(edges, points, queries, bipartite, output_filename, distances_filename);

    return 0;
}