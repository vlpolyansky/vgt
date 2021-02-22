#pragma once

#include <vector>
#include <cassert>
#include <cmath>
#include <string>
#include <memory>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include <Eigen/Dense>
#include <cnpy.h>

#ifdef _OPENMP
#include <omp.h>
#else
int omp_get_num_threads() {
    return 1;
}
int omp_get_thread_num() {
    return 0;
}
void omp_set_num_threads(int num_threads) { }
#endif


#ifndef FTYPE
#define FTYPE long double
#endif

using ftype = FTYPE;

#ifndef DIMENSIONALITY
#define DIMENSIONALITY Eigen::Dynamic
#endif

template<int delta>
constexpr int dim_delta() {
    return DIMENSIONALITY == Eigen::Dynamic ? DIMENSIONALITY : DIMENSIONALITY + delta;
}

using dmatrix = Eigen::Matrix<ftype, DIMENSIONALITY, Eigen::Dynamic, Eigen::ColMajor>;
using dvector = Eigen::Matrix<ftype, DIMENSIONALITY, 1>;   // "data" vector
using svector = Eigen::Matrix<ftype, dim_delta<1>(), 1>;   // "simplicial" vector


#define PI 3.141592653589793238462643383279
#define PI_f 3.141592653589793238462643383279f

template<class K, class V>
using map_t = std::unordered_map<K, V>;

template<class K>
using set_t = std::unordered_set<K>;

template<typename T>
using vec = std::vector<T>;

template<class T>
using ptr = std::shared_ptr<T>;

template<class T>
using uptr = std::unique_ptr<T>;

void ensure(bool condition, const std::string &error_message);

void omp_optimize_num_threads(int niter);

ftype sqr(ftype x);

int parse_int(const std::string &s);

ftype parse_float(const std::string &s);

dmatrix npy2matrix(const cnpy::NpyArray &array);


// -- Alternative to argparse --
char* get_cmd_option(char **begin, char **end, const std::string &option);
float get_cmd_option_float(char **begin, char **end, const std::string &option, float _default = 0.0);
int get_cmd_option_int(char **begin, char **end, const std::string &option, int _default = 0);
std::string get_cmd_option_string(char **begin, char **end, const std::string &option, const std::string &_default = "");
bool cmd_option_exists(char **begin, char **end, const std::string &option);
