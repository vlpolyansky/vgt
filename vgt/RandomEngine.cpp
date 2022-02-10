
#include "RandomEngine.h"

RandomEngine::RandomEngine(int seed) : re(seed) {}

int RandomEngine::rand_int(int n) {
    return std::uniform_int_distribution<int>(0, n - 1)(re);
}

ftype RandomEngine::rand_float() {
    return static_cast<ftype>(re() - RandomEngineType::min()) /
           static_cast<ftype>(RandomEngineType::max() - RandomEngineType::min());
}

ftype RandomEngine::rand_normal() {
    // return normal_distribution(random_engine);
    // Marsaglia polar method
    if (has_next_rand_normal) {
        has_next_rand_normal = false;
        return next_rand_normal;
    }
    ftype u, v, s;
    do {
        u = rand_float() * 2.0f - 1.0f;
        v = rand_float() * 2.0f - 1.0f;
        s = u * u + v * v;
    } while (s == 0 || s >= 1);
    ftype t = math::sqrt(-2.0f * math::log(s) / s);
    has_next_rand_normal = true;
    next_rand_normal = v * t;
    return u * t;
}

dvector RandomEngine::rand_on_sphere(int ndim) {
    dvector a(ndim);
    for (int i = 0; i < ndim; i++) {
        a[i] = rand_normal();
    }
    a.normalize();
    return a;
}

RandomEngine::RandomEngineType& RandomEngine::generator() {
    return re;
}

RandomEngineMultithread::RandomEngineMultithread(int seed) : seed(seed) {
    engines.emplace_back(seed);
}

void RandomEngineMultithread::fix_random_engines() {
    #pragma omp master
    {
        int num = omp_get_num_threads();
        for (int i = engines.size(); i < num; i++) {
            engines.emplace_back((1 + i) * seed);
        }
    }
    #pragma omp barrier
}

RandomEngine& RandomEngineMultithread::current() {
    int idx = omp_get_thread_num();
    ensure(idx < engines.size(), "Need to `fix_random_engines()` during multithreading");
    return engines[idx];
}

RandomEngineMultithread::RandomEngineMultithread(const RandomEngine &engine) {
    engines = {engine};
    seed = 239;
}
