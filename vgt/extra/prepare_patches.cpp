#include <random>
#include <algorithm>
#include <sstream>

#include "cnpy.h"

#include "../utils.h"

float dist(const float *a, const float *b, int dim) {
    float ret = 0;
    for (int i = 0; i < dim; i++) {
        float t = a[i] - b[i];
        ret += t * t;
    }
    return ret;
}

int main(int argc, char **argv) {
    std::string filename = argv[1];
    cnpy::NpyArray data_npy = cnpy::npy_load(filename);
    ensure(data_npy.shape.size() == 2, "Data should be represented as a matrix.");
    ensure(data_npy.word_size == sizeof(float), "Data word size should be 32 bit.");

    int N = data_npy.shape[0];
    size_t dim = data_npy.shape[1];

    int n = get_cmd_option_int(argv, argv + argc, "-n", N);  // 50000
    int k = get_cmd_option_int(argv, argv + argc, "-k", n / 500);  // 300
    int t = get_cmd_option_int(argv, argv + argc, "-t", 30);

    float sigma = get_cmd_option_float(argv, argv + argc, "-s", 0);

    n = std::min(n, N);

    vec<int> indices(N);
    for (int i = 0; i < N; i++) {
        indices[i] = i;
    }

    int seed = get_cmd_option_int(argv, argv + argc, "--seed", 239);

    // pick random `n` points
    std::cout << "Selecting " << n << " random points" << std::endl;
    std::default_random_engine generator(seed);
    for (int i = 0; i < n; i++) {
        int j = std::uniform_int_distribution<int>(i, N - 1)(generator);
        std::swap(indices[i], indices[j]);
    }

    float *data_npy_begin = data_npy.data<float>();

    auto noise = std::normal_distribution<float>(0, sigma);

    vec<float> data(n * dim);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < dim; j++) {
            data[i * dim + j] = data_npy_begin[indices[i] * dim  + j] + noise(generator);
        }
    }

    float *data_begin = data.data();

    std::cout << "Computing distance matrix" << std::endl;
    vec<vec<float>> sqr_dst(n, vec<float>(n, 0));
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            sqr_dst[i][j] = sqr_dst[j][i] = dist(data_begin + i * dim, data_begin + j * dim, dim);
        }
    }

    std::cout << "Determining " << k << "th neighbors" << std::endl;
    vec<float> kth(n, 0);
    for (int i = 0; i < n; i++) {
        vec<int> tmp(n);
        for (int j = 0; j < n; j++) {
            tmp[j] = j;
        }
        std::nth_element(tmp.begin(), tmp.begin() + k, tmp.end(), [&](int a, int b) {
            return sqr_dst[i][a] < sqr_dst[i][b];
        });
        kth[i] = tmp[k];
    }

    std::cout << "Determining the " << t << "% threshold for delta_k" << std::endl;
    vec<int> tmp(n);
    for (int i = 0; i < n; i++) {
        tmp[i] = i;
    }
    int t_to_idx = n * t / 100;
    std::nth_element(tmp.begin(), tmp.begin() + t_to_idx, tmp.end(), [&](int a, int b) {
        return sqr_dst[a][kth[a]] < sqr_dst[b][kth[b]];
    });
    float threshold = sqr_dst[tmp[t_to_idx]][kth[tmp[t_to_idx]]];

    std::cout << "Sampling the resulting points (" << threshold << ")" << std::endl;
    vec<int> final_indices;
    for (int i = 0; i < n; i++) {
        if (sqr_dst[i][kth[i]] <= threshold) {
            final_indices.push_back(i);
        }
    }
    vec<float> new_data(final_indices.size() * dim);
    for (size_t i = 0; i < final_indices.size(); i++) {
        std::copy(data_begin + final_indices[i] * dim, data_begin + (final_indices[i] + 1) * dim,
                new_data.begin() + i * dim);
    }

    assert(filename.size() > 4 && filename.substr(filename.size() - 4) == ".npy");
    std::stringstream out_filename;
    out_filename << filename.substr(0, filename.size() - 4)
            << "_n" << n << "_k" << k << "_t" << t << "_s" << sigma << ".npy";
    std::cout << "Saving to " << out_filename.str() << std::endl;
    cnpy::npy_save(out_filename.str(), new_data.data(), {final_indices.size(), dim});

    return 0;
}