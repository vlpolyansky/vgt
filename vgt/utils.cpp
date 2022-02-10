#include "utils.h"

#include <iostream>

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

ftype qpow(ftype a, int b) {
    if (b == 0) {
        return 1;
    } else if (b % 2 == 0) {
        return qpow(a * a, b / 2);
    } else {
        return a * qpow(a, b - 1);
    }
}

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

dvector dvector::col(const dmatrix &mat, int index) {
    dvector result = mat.col(index);
    result.index = index;
    return result;
}

const_dvector_ref const_dvector_ref::col(const dmatrix &mat, int index) {
    const_dvector_ref result = mat.col(index);
    result.index = index;
    return result;
}

dmatrix npy2matrix(const cnpy::NpyArray &data_npy) {
    ensure(data_npy.shape.size() == 2, "Data should be represented as a matrix.");
//    ensure(data_npy.word_size == sizeof(float), "Data word size should be 32 bit.");
    ensure(!data_npy.fortran_order, "C-order expected.");
    int n = data_npy.shape[0];
    int d = data_npy.shape[1];
    ensure(AMBIENT_DIM == Eigen::Dynamic || AMBIENT_DIM == d,
           "Program is compiled with AMBIENT_DIM=" + std::to_string(AMBIENT_DIM) +
           ", but input dimensionality is " + std::to_string(d));
    dmatrix data(d, n);
    if (data_npy.word_size == sizeof(float)) {
        std::transform(data_npy.data<float>(), data_npy.data<float>() + (n * d), data.data(),
                       [](float a) {return static_cast<ftype>(a);});
    } else if (data_npy.word_size == sizeof(double)) {
        std::transform(data_npy.data<double>(), data_npy.data<double>() + (n * d), data.data(),
                       [](double a) {return static_cast<ftype>(a);});
    } else {
        throw std::runtime_error("Unknown data word size.");
    }
//    for (int i = 0; i < n; i++) {
//        for (int j = 0; j < d; j++) {
//            data(j, i) = data_npy.data<float>()[i * d + j];
//        }
//    }
    return data;
}

my_tqdm::my_tqdm(int max_value) : _bar(), _max_value(max_value), _counter(0) {
    _bar.disable_colors();
    _bar.set_width(10);
}

tqdm &my_tqdm::bar() {
    return _bar;
}

int &my_tqdm::max_value() {
    return _max_value;
}

void my_tqdm::atomic_iteration() {
    #pragma omp critical
    _bar.progress(_counter++, _max_value);
}

ftype nball_volume(int n) {
    static vec<ftype> a = {1.0f, 2.0f};
    while (a.size() < n + 1) {
        a.push_back(a[a.size() - 2] * 2 * PI_ftype / static_cast<ftype>(a.size()));
    }
    return a[n];
}

ftype factorial(int n) {
    static vec<ftype> a = {static_cast<ftype>(1)};
    while (a.size() < n + 1) {
        int k = a.size();
        a.push_back(a[k - 1] * static_cast<ftype>(k));
    }
    return a[n];
}

ftype nsphere_volume(int n) {
    return nball_volume(n + 1) / (n + 1);
}

/**
 * O(1) computation for each series term.
 */
ftype __solid_angle_term_2(const vec<ftype> &alpha, const vec<std::pair<int, int>> &meshgrid, vec<int> &index,
                           int cur_index_sum, int last_index_added, int max_index_sum,
                           ftype term_11, ftype term_12,
        /*sum(a_im)*/vec<ftype> &term_2i112, vec<ftype> &term_2i, ftype term_2,
                           ftype term_3) {
    // compute current term
    ftype result = term_11 / term_12 * term_2 * term_3;

    // descent the multi-index tree
    if (cur_index_sum < max_index_sum) {
        for (int k = last_index_added; k < index.size(); k++) {
            index[k]++;
            int i = meshgrid[k].first;
            int j = meshgrid[k].second;
            term_2i112[i]++;
            term_2i112[j]++;
            ftype old_term_2i_i = term_2i[i];
            ftype old_term_2i_j = term_2i[j];
            term_2i[i] = tgamma(static_cast<ftype>(0.5) * (1 + term_2i112[i]));
            term_2i[j] = tgamma(static_cast<ftype>(0.5) * (1 + term_2i112[j]));
            ftype new_term_2 = term_2 * (term_2i[i] * term_2i[j] / (old_term_2i_i * old_term_2i_j));
            result += __solid_angle_term_2(alpha, meshgrid, index,
                                           cur_index_sum + 1, k, max_index_sum,
                                           term_11 * (-2), term_12 * index[k],
                                           term_2i112, term_2i, new_term_2,
                                           term_3 * alpha[k]);
            index[k]--;
            term_2i112[i]--;
            term_2i112[j]--;
            term_2i[i] = old_term_2i_i;
            term_2i[j] = old_term_2i_j;
        }
    }

    return result;
}

ftype solid_angle(const Eigen::Matrix<ftype, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> &V, int taylor_degree) {
    // !!Assuming that vectors are normalized!!
    // Ribando's taylor series formula
    int n = V.cols();
    if (n == 2) {
        return acos(V.col(0).dot(V.col(1)));
    }
    Eigen::Matrix<ftype, Eigen::Dynamic, Eigen::Dynamic> A = V.transpose() * V;
    vec<ftype> alpha;
    vec<std::pair<int, int>> meshgrid;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            alpha.push_back(A(i, j));
            meshgrid.push_back(std::make_pair(i, j));
        }
    }
    vec<int> index(alpha.size(), 0);
    ftype term_1 = abs(V.determinant()) / pow(4 * PI_ftype, static_cast<ftype>(0.5) * n);

    vec<ftype> term_22i112(n, 0);
    vec<ftype> term_22i(n, tgamma(static_cast<ftype>(0.5)));
    ftype term_2 = __solid_angle_term_2(alpha, meshgrid, index,
                                        0, 0, taylor_degree,
                                        1, 1,
                                        term_22i112, term_22i, qpow(term_22i[0], n),
                                        1);

    return term_1 * term_2;
}

ftype simplex_volume(const Eigen::Matrix<ftype, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> &V) {
    return V.determinant() / factorial(V.cols());
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
