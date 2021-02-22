#pragma once

#include <random>

#include "utils.h"

class RandomEngine {
public:
    explicit RandomEngine(int seed = 239);

    int rand_int(int n);
    ftype rand_float();
    ftype rand_normal();
    dvector rand_on_sphere(int ndim);

private:
    using RandomEngineType = std::mt19937;
    RandomEngineType re;

    bool has_next_rand_normal = false;
    ftype next_rand_normal;
};

class RandomEngineMultithread {
public:
    RandomEngineMultithread(int seed);

    void fix_random_engines();

    RandomEngine& current();

private:
    int seed;
    vec<RandomEngine> engines;
};