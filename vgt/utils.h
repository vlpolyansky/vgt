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
#include <tqdm.h>

#ifdef _OPENMP
#include <omp.h>

#else
int omp_get_num_threads();
int omp_get_thread_num();
void omp_set_num_threads(int num_threads);
#endif

//#define MPC

#ifdef MPC
//#include <boost/math/bindings/mpfr.hpp>
#include <boost/multiprecision/mpfr.hpp>
namespace math = boost::multiprecision;
#else
namespace math = std;
#endif


#ifndef FTYPE
#ifdef MPC
#define FTYPE math::mpf_float_1000
#else
#define FTYPE long double
#endif
#endif

using ftype = FTYPE;

template<int dim, int delta>
constexpr int dim_delta() {
    return dim == Eigen::Dynamic ? dim : dim + delta;
}

// todo update documentation
#ifndef AMBIENT_DIM
#define AMBIENT_DIM Eigen::Dynamic
#endif

#ifndef DATA_DIM
#define DATA_DIM Eigen::Dynamic // AMBIENT_DIM
#endif


using dynmatrix  = Eigen::Matrix<ftype, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using dynvector = Eigen::Matrix<ftype, Eigen::Dynamic, 1>;

using dmatrix  = Eigen::Matrix<ftype, AMBIENT_DIM, Eigen::Dynamic, Eigen::ColMajor>;
using const_dmatrix_ref  = Eigen::Ref<const dmatrix>;
using dvector_base = Eigen::Matrix<ftype, AMBIENT_DIM, 1>;  // "data" [d] vector
using const_dvector_base_ref = Eigen::Ref<const dvector_base>;
class dvector : public dvector_base {
public:
    using dvector_base::dvector_base;
    int index = -1;
    /** Create a dvector from a matrix column, storing its column index. */
    static dvector col(const dmatrix &mat, int index);
};
class const_dvector_ref : public const_dvector_base_ref {
public:
    using const_dvector_base_ref::const_dvector_base_ref;
    int index = -1;
    /** Create a dvector from a matrix column, storing its column index. */
    static const_dvector_ref col(const dmatrix &mat, int index);
};
//using const_dvector_ref = dvector;
using svector = Eigen::Matrix<ftype, dim_delta<DATA_DIM, 1>(), 1>;   // "simplicial" [d+1] vector


#define PI_ftype (static_cast<ftype>(3.141592653589793238462643383279))
#define INF_ftype (std::numeric_limits<ftype>::infinity())
#define NAN_ftype (std::numeric_limits<ftype>::quiet_NaN())

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

ftype qpow(ftype a, int b);

void ensure(bool condition, const std::string &error_message);

void omp_optimize_num_threads(int niter);

ftype sqr(ftype x);

int parse_int(const std::string &s);

ftype parse_float(const std::string &s);

dmatrix npy2matrix(const cnpy::NpyArray &array);

// -- tqdm --

class my_tqdm {
public:
    explicit my_tqdm(int max_value);
    tqdm& bar();
    int& max_value();
    void atomic_iteration();
private:
    tqdm _bar;
    int _counter;
    int _max_value;
};

// -- Geometry --
ftype nball_volume(int n);
ftype nsphere_volume(int n);
ftype solid_angle(const Eigen::Matrix<ftype, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> &V, int taylor_degree=1);
ftype simplex_volume(const Eigen::Matrix<ftype, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> &V/*n^2*/);


// -- Alternative to argparse --
char* get_cmd_option(char **begin, char **end, const std::string &option);
float get_cmd_option_float(char **begin, char **end, const std::string &option, float _default = 0.0);
int get_cmd_option_int(char **begin, char **end, const std::string &option, int _default = 0);
std::string get_cmd_option_string(char **begin, char **end, const std::string &option, const std::string &_default = "");
bool cmd_option_exists(char **begin, char **end, const std::string &option);
