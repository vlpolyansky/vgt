#include <memory>
#include <fstream>
#include <iostream>
#include <tqdm.h>
#include <argparse.hpp>

#include "DelaunayApproximator.h"
#include "VoronoiGraph.h"

DelaunayApproximator::DelaunayApproximator(int seed, const VoronoiGraph &graph, int delaunay_dim) :
        re(seed), graph(graph), delaunay_dim(delaunay_dim), n(graph.get_data_size()), d(graph.get_data_dim()) {
}

void DelaunayApproximator::initialize_vertices() {
    initial_vertex_vec = vec<Polytope>(n, VoronoiGraph::NONE);
    #pragma omp parallel
    {
        re.fix_random_engines();
        #pragma omp for
        for (int i = 0; i < n; i++) {
            initial_vertex_vec[i] = graph.retrieve_vertex_nearby(i, re.current());
        }
    }
}

void DelaunayApproximator::random_walk(int n_steps, bool save_vertices) {
    tqdm bar;
    bar.set_width(10);
    bar.disable_colors();
    int bar_counter = 0;
    int bar_max = initial_vertex_vec.size();

    #pragma omp parallel
    {
        re.fix_random_engines();

        #pragma omp for
        for (int i = 0; i < initial_vertex_vec.size(); i++) {
            #pragma omp critical
            bar.progress(bar_counter++, bar_max);

            Polytope vertex = initial_vertex_vec[i];
            if (vertex.is_none()) {
                continue;
            }
            set_t<IndexSet> local_skeleton;
            set_t<IndexSet> local_vertices;
            add_delaunay_simplices(vertex.dual, local_skeleton);
            if (save_vertices) local_vertices.insert(vertex.dual);

            for (int step = 0; step < n_steps; step++) {
                Polytope next_vertex = graph.get_neighbor(vertex, re.current().rand_int(d + 1), re.current());
                if (!next_vertex.is_none()) { // make a step
                    add_delaunay_simplices(next_vertex.dual, local_skeleton);
                    if (save_vertices) local_vertices.insert(next_vertex.dual);
                    vertex = next_vertex;
                }
                if (local_skeleton.size() >= MAX_LOCAL_SIZE) { // update global set
                    #pragma omp critical
                    delaunay_skeleton.insert(local_skeleton.begin(), local_skeleton.end());

                    local_skeleton = set_t<IndexSet>();
                }
            }

            #pragma omp critical
            {
                delaunay_skeleton.insert(local_skeleton.begin(), local_skeleton.end());
                if (save_vertices) voronoi_vertices.insert(local_vertices.begin(), local_vertices.end());
            }
        }
    }
    bar.finish();

}

void DelaunayApproximator::full_walk() {
    tqdm bar;
    bar.set_width(10);
    bar.disable_colors();
    int bar_counter = 0;
    int bar_max = 0;

    vec<Polytope> wave = initial_vertex_vec;
    int wave_counter = 0;

    while (!wave.empty()) {
        bar.set_label("wave " + std::to_string(wave_counter++));
        bar_max += wave.size();

        vec<Polytope> new_wave;
        #pragma omp parallel
        {
            re.fix_random_engines();

            #pragma omp for
            for (int i = 0; i < wave.size(); i++) {
                #pragma omp critical
                bar.progress(bar_counter++, bar_max);
                const Polytope &vertex = wave[i];
                if (vertex.is_none()) {
                    continue;
                }
                vec<Polytope> neighbors = graph.get_neighbors(vertex, re.current());
                #pragma omp critical
                {
                    for (const Polytope &new_vertex: neighbors) {
                        if (!new_vertex.is_none() && voronoi_vertices.find(new_vertex.dual) == voronoi_vertices.end()) {
                            voronoi_vertices.insert(new_vertex.dual);
                            new_wave.push_back(new_vertex);
                        }
                    }
                }
            }
        }
    }
    bar.finish();
}

bool next_combination(vec<int> &a, int n, int k) {
    for (int i = k - 1; i >= 0; i--) {
        if (a[i] < n - k + i) {
            a[i]++;
            for (int j = i + 1; j < k; j++) {
                a[j] = a[j - 1] + 1;
            }
            return true;
        }
    }
    return false;
}

void DelaunayApproximator::add_delaunay_simplices(const IndexSet &simplex, set_t<IndexSet> &delaunay_set) {
    if (delaunay_dim >= 0 && simplex.dim() > delaunay_dim) {
        vec<int> a(delaunay_dim + 1);
        for (int i = 0; i < a.size(); i++) {
            a[i] = i;
        }
        vec<int> next_simplex(a.size());
        do {
            for (int i = 0; i < a.size(); i++) {
                next_simplex[i] = simplex[a[i]];
            }
            add_delaunay_simplices(next_simplex, delaunay_set);
        } while (next_combination(a, simplex.dim() + 1, delaunay_dim + 1));
    } else {
        if (delaunay_set.find(simplex) != delaunay_set.end()) {
            return;
        }
        if (compute_closure || delaunay_dim < 0 || simplex.dim() == delaunay_dim) {
            delaunay_set.insert(simplex);
        }
        if (compute_closure) {
            for (const IndexSet &next_simplex : simplex.boundary()) {
                add_delaunay_simplices(next_simplex, delaunay_set);
            }
        }
    }
}

void DelaunayApproximator::save_delaunay(const std::string &output_filename) {
    std::ofstream out(output_filename);
    for (const IndexSet &simplex: delaunay_skeleton) {
        out << simplex << '\n';
    }
    out.close();
}

void DelaunayApproximator::save_voronoi(const std::string &output_filename) {
    std::ofstream out(output_filename);
    for (const IndexSet &vertex: voronoi_vertices) {
        out << vertex << '\n';
    }
    out.close();
}

const VoronoiGraph &DelaunayApproximator::get_graph() const {
    return graph;
}

bool DelaunayApproximator::should_compute_closure() const {
    return compute_closure;
}

void DelaunayApproximator::set_compute_closure(bool compute_closure) {
    this->compute_closure = compute_closure;
}

argparse::ArgumentParser prepare_parser() {
    argparse::ArgumentParser program("Delaunay skeleton approximation");

    program.add_argument("data.npy")
            .help("npy NxD data matrix of 32-bit floats");

    program.add_argument("--dim")
            .help("maximum dimensionality of extracted Delaunay simplices")
            .default_value(2)
            .action(&parse_int);

    program.add_argument("--steps")
            .help("number of steps to perform in a random walk from each starting vertex;"
                  " a non-positive value would instead correspond to a full walk / complete graph search")
            .default_value(1000)
            .action(&parse_int);

    program.add_argument("--noclosure")
            .help("flag to output only k-dimensional Delaunay simplices without their closure")
            .default_value(false)
            .implicit_value(true);

    program.add_argument("--out")
            .help("output filename for "
                  "a txt file containing Delaunay simplices; each line describes"
                  " a simplex in a form \"d v_0 v_1 ... v_d [f]\"")
            .default_value(std::string("output.txt"));

    program.add_argument("--vout")
            .help("optional output filename for "
                  "a txt file containing visited Voronoi vertices; each line describes"
                  " a vertex in a form \"d v_0 v_1 ... v_d\"")
            .default_value(std::string(""));

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

    int seed = program.get<int>("--seed");
    auto strategy = program.get<RayStrategyType>("--strategy");
    std::string data_filename = program.get("data.npy");
    std::string delaunay_output_filename = program.get("--out");
    std::string vertices_output_filename = program.get("--vout");
    bool save_vertices = !vertices_output_filename.empty();
    int delaunay_dim = program.get<int>("--dim");
    int n_steps = program.get<int>("--steps");

    bool noclosure = program.get<bool>("--noclosure");

    // Initialize
    VoronoiGraph graph(strategy);
    std::cout << "Reading data points" << std::endl;
    graph.read_points(data_filename);
    DelaunayApproximator walk(seed, graph, delaunay_dim);
    walk.set_compute_closure(!noclosure);

    // Stage 1 - initial set of vertices
    std::cout << "Initializing starting vertices" << std::endl;
    walk.initialize_vertices();

    // Stage 2 - walk
    if (n_steps > 0) {
        std::cout << "Performing random walk with " << n_steps << " steps" << std::endl;
        walk.random_walk(n_steps, save_vertices);
    } else {
        std::cout << "Performing full walk" << std::endl;
        walk.full_walk();
    }

    graph.print_validations_info();

    walk.save_delaunay(delaunay_output_filename);
    if (save_vertices) {
        walk.save_voronoi(vertices_output_filename);
    }

    return 0;
}