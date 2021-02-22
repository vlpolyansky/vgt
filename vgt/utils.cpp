#include "utils.h"

#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

void ensure(bool condition, const std::string &error_message) {
    if (!condition) {
        throw std::runtime_error(error_message);
    }
}

void omp_optimize_num_threads(int niter) {
#ifdef _OPENMP
    int num_proc = omp_get_num_procs();
    int min_blocks = (niter + num_proc - 1) / num_proc;
    int min_num_threads = (niter + min_blocks - 1) / min_blocks;
    min_num_threads = std::min(num_proc, min_num_threads);
    std::cout << "Using " << min_num_threads << " threads" << std::endl;
    omp_set_num_threads(min_num_threads);
#endif
}

ftype sqr(ftype x) {
    return x * x;
}

int parse_int(const std::string &s) {
    return std::stoi(s);
}

ftype parse_float(const std::string &s) {
    return static_cast<ftype>(std::stof(s));
}

dmatrix npy2matrix(const cnpy::NpyArray &data_npy) {
    ensure(data_npy.shape.size() == 2, "Data should be represented as a matrix.");
    ensure(data_npy.word_size == sizeof(float), "Data word size should be 32 bit.");
    int n = data_npy.shape[0];
    int d = data_npy.shape[1];
    ensure(DIMENSIONALITY == Eigen::Dynamic || DIMENSIONALITY == d,
           "Program is compiled with _DIMENSIONALITY=" + std::to_string(DIMENSIONALITY) +
           ", but data dimensionality is " + std::to_string(d));
    dmatrix data(d, n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) {
            data(j, i) = data_npy.data<float>()[i * d + j];
        }
    }
    return data;
}

char* get_cmd_option(char **begin, char **end, const std::string &option) {
    char **itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return nullptr;
}

bool cmd_option_exists(char **begin, char **end, const std::string &option) {
    return std::find(begin, end, option) != end;
}

float get_cmd_option_float(char **begin, char **end, const std::string &option, float _default) {
    char *result = get_cmd_option(begin, end, option);
    return result ? static_cast<float>(std::atof(result)) : _default;
}

int get_cmd_option_int(char **begin, char **end, const std::string &option, int _default) {
    char *result = get_cmd_option(begin, end, option);
    return result ? std::atoi(result) : _default;
}

std::string get_cmd_option_string(char **begin, char **end, const std::string &option, const std::string &_default) {
    char *result = get_cmd_option(begin, end, option);
    return result ? std::string(result) : _default;
}
