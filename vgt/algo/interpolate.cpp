
#include <cnpy.h>
#include <argparse.hpp>
#include <tqdm.h>

#include "../utils.h"
#include "VoronoiGraph.h"

argparse::ArgumentParser prepare_parser() {
    argparse::ArgumentParser program("Piecewise-linear interpolation");

    program.add_argument("data.npy")
            .help("npy NxD data matrix of 32-bit floats");

    program.add_argument("query.npy")
            .help("npy MxD query matrix of 32-bit floats");

    program.add_argument("--values")
            .help("npy Nx1 matrix of 32-bit floats")
            .default_value(std::string(""));

    program.add_argument("--out")
            .help("output filename for "
                  "an npz data file containing 'indices', 'simplices', 'coefficients' and"
                  " (if 'values' are provided) 'estimates' matrices")
            .default_value(std::string("interpolated.npz"));

    program.add_argument("--seed")
            .help("random seed")
            .default_value(239)
            .action(&parse_int);

    program.add_argument("--njobs")
            .help("number of parallel threads (requires OpenMP)")
            .default_value(1)
            .action(&parse_int);

    program.add_argument("--strategy")
            .help("ray sampling strategy from {'brute_force', 'bin_search'}")
            .default_value(BRUTE_FORCE)
            .action([](const std::string &s) {
                if (s == "brute_force") {
                    return BRUTE_FORCE;
                } else if (s == "bin_search") {
                    return BIN_SEARCH;
                } else {
                    throw std::runtime_error("unknown ray casting strategy: " + s);
                }
            });
    return program;
}

int main(int argc, char **argv) {
    argparse::ArgumentParser program = prepare_parser();
    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cout << err.what() << std::endl;
        std::cout << program;
        exit(0);
    }

    omp_set_num_threads(program.get<int>("--njobs"));

    std::string data_filename = program.get("data.npy");
    std::string query_filename = program.get("query.npy");

    int seed = program.get<int>("--seed");
    auto strategy = program.get<RayStrategyType>("--strategy");

    std::string values_filename = program.get("--values");
    bool save_values = !values_filename.empty();
    std::string output_filename = program.get("--out");

    RandomEngineMultithread re(seed);

    // Initialize
    VoronoiGraph graph(strategy);
    std::cout << "Reading data" << std::endl;
    graph.read_points(data_filename);

    std::cout << "Reading query points" << std::endl;
    cnpy::NpyArray query_npy = cnpy::npy_load(query_filename);
    ensure(query_npy.shape.size() == 2, "Data should be represented as a matrix.");
    ensure(query_npy.word_size == sizeof(float), "Data word size should be 32 bit.");

    dmatrix queries = npy2matrix(query_npy);
    size_t n_queries = queries.cols();
    size_t dim = queries.rows();
    ensure(dim == graph.get_data_dim(), "Points have different dimensions");

    vec<ftype> values;

    if (save_values) {
        std::cout << "Reading function values" << std::endl;
        cnpy::NpyArray values_npy = cnpy::npy_load(values_filename);
        ensure(values_npy.shape.size() == 2, "Data should be represented as a matrix.");
        ensure(values_npy.word_size == sizeof(float), "Data word size should be 32 bit.");
        ensure(values_npy.shape[0] == graph.get_data_size(), "Number of values is different"
                                                             " from the size of data.");
        ensure(values_npy.shape[1] == 1, "values should have size Nx1");
        vec<float> values_float = values_npy.as_vec<float>();
        values = vec<ftype>(values_float.begin(), values_float.end());
    }

    vec<int> all_indices;
    vec<int> all_simplices;
    vec<ftype> all_coefficients;
    vec<ftype> all_estimates;

    tqdm bar;
    bar.disable_colors();
    bar.set_width(10);
    int bar_counter = 0;

    std::cout << "Computing the interpolation" << std::endl;
    #pragma omp parallel
    {
        re.fix_random_engines();

        #pragma omp for
        for (int i = 0; i < n_queries; i++) {
            #pragma omp critical
            bar.progress(bar_counter++, n_queries);
            svector coefficients;
            VoronoiGraph::Polytope vertex = graph.retrieve_vertex(queries.col(i), re.current(),
                                                                  &coefficients);
            if (!vertex.is_none()) {
                ftype estimate = 0;
                if (save_values) {
                    for (int j = 0; j < coefficients.size(); j++) {
                        estimate += coefficients[j] * values[vertex.dual[j]];
                    }
                }
                #pragma omp critical
                {
                    all_indices.push_back(i);
                    all_simplices.insert(all_simplices.end(), vertex.dual.begin(), vertex.dual.end());
                    all_coefficients.insert(all_coefficients.end(),
                                            coefficients.data(), coefficients.data() + coefficients.size());
                    all_estimates.push_back(estimate);
                }
            }

        }
    }
    bar.finish();

    graph.print_validations_info();

    std::cout << "Saving results" << std::endl;
    size_t nq = all_indices.size();
    cnpy::npz_save(output_filename, "indices", all_indices.data(), {nq}, "w");
    cnpy::npz_save(output_filename, "simplices", all_simplices.data(), {nq, dim + 1}, "a");
    cnpy::npz_save(output_filename, "coefficients", all_coefficients.data(), {nq, dim + 1}, "a");
    cnpy::npz_save(output_filename, "estimates", all_estimates.data(), {nq}, "a");
}