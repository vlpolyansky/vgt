#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#include <cfloat>
#include <cstdio>
#include <cmath>
#endif

#line 7

#define uint unsigned int

#ifndef UPDATE_DATA_SQUARED_GS0
#define UPDATE_DATA_SQUARED_GS0 1
#endif
#ifndef UPDATE_DATA_SQUARED_GS1
#define UPDATE_DATA_SQUARED_GS1 1
#endif
__kernel void update_data_squared(__global const float *data,
                                  __global       float *data_squared,
                                  int data_n, int dim) {
    const uint data_idx = get_global_id(1);
    const uint loc_0 = get_local_id(0);  // dim
    const uint loc_1 = get_local_id(1);  // data_n

    float sum = 0.0f;
    float t;

    if (data_idx < data_n) {
        for (int i = loc_0; i < dim; i += UPDATE_DATA_SQUARED_GS0) {
            t = data[data_idx * dim + i];
            sum += t * t;
        }
    }

    __local float sum_local[UPDATE_DATA_SQUARED_GS0][UPDATE_DATA_SQUARED_GS1];
    sum_local[loc_0][loc_1] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (data_idx < data_n && loc_0 == 0) {
        sum = 0.0f;
        for (int i = 0; i < UPDATE_DATA_SQUARED_GS0; i++) {
            sum += sum_local[i][loc_1];
        }
        data_squared[data_idx] = sum;
    }
}


#ifndef MATMUL_GS0
#define MATMUL_GS0 1
#endif
#ifndef MATMUL_GS1
#define MATMUL_GS1 1
#endif
#ifndef MATMUL_GS2
#define MATMUL_GS2 1
#endif
//  (a_size, dim) * (b_size, dim)^t
__kernel void matmul(__global const float *a,
                     __global const float *b_t,
                     __global       float *c,
                     int a_size, int b_size, int dim) {
    const uint idx1 = get_global_id(1);
    const uint idx2 = get_global_id(2);

    const uint loc_0 = get_local_id(0); // dim
    const uint loc_1 = get_local_id(1); // a_size
    const uint loc_2 = get_local_id(2); // b_size

    float sum = 0.0f;
    float t;

    if (idx1 < a_size && idx2 < b_size) {
        for (int i = loc_0; i < dim; i += MATMUL_GS0) {
            t = a[idx1 * dim + i] * b_t[idx2 * dim + i];
            sum += t;
        }
    }

    __local float sum_local[MATMUL_GS0][MATMUL_GS1][MATMUL_GS2];
    sum_local[loc_0][loc_1][loc_2] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (idx1 < a_size && idx2 < b_size && loc_0 == 0) {
        sum = 0.0f;
        for (int i = 0; i < MATMUL_GS0; i++) {
            sum += sum_local[i][loc_1][loc_2];
        }
        c[idx1 * b_size + idx2] = sum;
    }
}


#ifndef INTERSECT_RAY_BRUTEFORCE_GS0
#define INTERSECT_RAY_BRUTEFORCE_GS0 1
#endif
#ifndef INTERSECT_RAY_BRUTEFORCE_GS1
#define INTERSECT_RAY_BRUTEFORCE_GS1 1
#endif
#ifndef INTERSECT_RAY_BRUTEFORCE_GS2
#define INTERSECT_RAY_BRUTEFORCE_GS2 1
#endif
__kernel void intersect_ray_bruteforce(__global const float *data_squared,
                                       __global const float *ref_squared,
                                       __global const float *ref_dot_data,
                                       __global const float *u_dot_data,
                                       __global const float *u_dot_ref,
                                       __global const int   *j0,
                                       __global       int   *best_j_pos,
                                       __global       int   *best_j_neg,
                                       int data_n, int ref_n, int u_n) {
    // ti = (p0^2 - pj^2 + 2ref_dot_pj - 2ref_dot_p0) / (2u_dot_pj - 2u_dot_p0)
    const uint ref_i = get_global_id(1);
    const uint u_i = get_global_id(2);

    const uint loc_0 = get_local_id(0); // data_n
    const uint loc_1 = get_local_id(1); // ref_n
    const uint loc_2 = get_local_id(2); // u_n

    int best_j_pos_single = -1;
    float best_t_pos_single = FLT_MAX;
    int best_j_neg_single = -1;
    float best_t_neg_single = -FLT_MAX;

    int j0_current = -1;
    float p0_squared = 0.0f;
    float ref_dot_p0 = 0.0f;
    float u_dot_p0 = 0.0f;

    if (ref_i < ref_n && u_i < u_n) {
        j0_current = j0[ref_i];
        if (j0_current >= 0) {
            p0_squared = data_squared[j0_current];
            ref_dot_p0 = ref_dot_data[ref_i * data_n + j0_current];
            u_dot_p0 = u_dot_data[u_i * data_n + j0_current];
        } else {
            p0_squared = ref_squared[ref_i];
            ref_dot_p0 = p0_squared;
            u_dot_p0 = u_dot_ref[u_i * ref_n + ref_i];
        }
    }

//    __local float pi_squared[INTERSECT_RAY_BRUTEFORCE_GS0][INTERSECT_RAY_BRUTEFORCE_GS1][INTERSECT_RAY_BRUTEFORCE_GS2];
//    __local float ref_dot_pi[INTERSECT_RAY_BRUTEFORCE_GS0][INTERSECT_RAY_BRUTEFORCE_GS1][INTERSECT_RAY_BRUTEFORCE_GS2];
//    __local float u_dot_pi[INTERSECT_RAY_BRUTEFORCE_GS0][INTERSECT_RAY_BRUTEFORCE_GS1][INTERSECT_RAY_BRUTEFORCE_GS2];

    if (ref_i < ref_n && u_i < u_n) {
        for (int j = loc_0; j < data_n; j += INTERSECT_RAY_BRUTEFORCE_GS0) {
            if (j == j0_current) {
                continue;
            }
            float pj_squared = data_squared[j];
            float ref_dot_pj = ref_dot_data[ref_i * data_n + j];
            float u_dot_pj = u_dot_data[u_i * data_n + j];

            float tj = (p0_squared - pj_squared - 2 * (ref_dot_p0 - ref_dot_pj)) / (2 * (u_dot_p0 - u_dot_pj));
            if (isnan(tj) || isinf(tj)) {
                continue;
            }

            if (u_dot_pj > u_dot_p0) {
                if (best_j_pos_single < 0 || best_t_pos_single > tj) {
                    best_j_pos_single = j;
                    best_t_pos_single = tj;
                }
            } else {
                if (best_j_neg_single < 0 || best_t_neg_single < tj) {
                    best_j_neg_single = j;
                    best_t_neg_single = tj;
                }
            }
        }
    }

    __local int best_j_pos_local[INTERSECT_RAY_BRUTEFORCE_GS0][INTERSECT_RAY_BRUTEFORCE_GS1][INTERSECT_RAY_BRUTEFORCE_GS2];
    __local float best_t_pos_local[INTERSECT_RAY_BRUTEFORCE_GS0][INTERSECT_RAY_BRUTEFORCE_GS1][INTERSECT_RAY_BRUTEFORCE_GS2];
    __local int best_j_neg_local[INTERSECT_RAY_BRUTEFORCE_GS0][INTERSECT_RAY_BRUTEFORCE_GS1][INTERSECT_RAY_BRUTEFORCE_GS2];
    __local float best_t_neg_local[INTERSECT_RAY_BRUTEFORCE_GS0][INTERSECT_RAY_BRUTEFORCE_GS1][INTERSECT_RAY_BRUTEFORCE_GS2];
    best_j_pos_local[loc_0][loc_1][loc_2] = best_j_pos_single;
    best_t_pos_local[loc_0][loc_1][loc_2] = best_t_pos_single;
    best_j_neg_local[loc_0][loc_1][loc_2] = best_j_neg_single;
    best_t_neg_local[loc_0][loc_1][loc_2] = best_t_neg_single;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (ref_i < ref_n && u_i < u_n && loc_0 == 0) {
        best_j_pos_single = -1;
        best_t_pos_single = FLT_MAX;
        best_j_neg_single = -1;
        best_t_neg_single = -FLT_MAX;
        for (int i = 0; i < INTERSECT_RAY_BRUTEFORCE_GS0; i++) {
            if (best_j_pos_single < 0 || best_t_pos_single > best_t_pos_local[i][loc_1][loc_2]) {
                best_j_pos_single = best_j_pos_local[i][loc_1][loc_2];
                best_t_pos_single = best_t_pos_local[i][loc_1][loc_2];
            }
            if (best_j_neg_single < 0 || best_t_neg_single < best_t_neg_local[i][loc_1][loc_2]) {
                best_j_neg_single = best_j_neg_local[i][loc_1][loc_2];
                best_t_neg_single = best_t_neg_local[i][loc_1][loc_2];
            }
        }
        best_j_pos[ref_i * u_n + u_i] = best_j_pos_single;
        best_j_neg[ref_i * u_n + u_i] = best_j_neg_single;
    }
}


#ifndef INTERSECT_RAY_INDEXED_GS0
#define INTERSECT_RAY_INDEXED_GS0 1
#endif
#ifndef INTERSECT_RAY_INDEXED_GS1
#define INTERSECT_RAY_INDEXED_GS1 1
#endif
__kernel void intersect_ray_indexed(__global const float *data_squared,
                                    __global const float *ref_dot_data,
                                    __global const float *u_dot_data,
                                    __global const int   *j0,  // sorted locally, split with -1
                                    __global const int   *u_indices,
                                    __global       int   *best_j_pos,
                                    __global       int   *best_j_neg,
                                    int data_n, int ref_n, int u_n) {
    // ti = (p0^2 - pj^2 + 2ref_dot_pj - 2ref_dot_p0) / (2u_dot_p0 - 2u_dot_pj)
    const uint ref_i = get_global_id(1);

    const uint loc_0 = get_local_id(0); // data_n
    const uint loc_1 = get_local_id(1); // ref_n

    int best_j_pos_single = -1;
    float best_t_pos_single = FLT_MAX;
    int best_j_neg_single = -1;
    float best_t_neg_single = -FLT_MAX;

    int u_i = -1;
    int j0_current = -1;
    float p0_squared = 0.0f;
    float ref_dot_p0 = 0.0f;
    float u_dot_p0 = 0.0f;

    if (ref_i < ref_n) {
        u_i = u_indices[ref_i];
        j0_current = j0[ref_i];
        p0_squared = data_squared[j0_current];
        ref_dot_p0 = ref_dot_data[ref_i * data_n + j0_current];
        u_dot_p0 = u_dot_data[u_i * data_n + j0_current];
    }

    if (ref_i < ref_n) {
        for (int j = loc_0; j < data_n; j += INTERSECT_RAY_INDEXED_GS0) {
            if (j == j0_current) {
                continue;
            }
            float pj_squared = data_squared[j];
            float ref_dot_pj = ref_dot_data[ref_i * data_n + j];
            float u_dot_pj = u_dot_data[u_i * data_n + j];

            float tj = (p0_squared - pj_squared - 2 * (ref_dot_p0 - ref_dot_pj)) / (2 * (u_dot_p0 - u_dot_pj));
            if (isnan(tj) || isinf(tj)) {
                continue;
            }

            if (u_dot_pj > u_dot_p0) {
                if (best_j_pos_single < 0 || best_t_pos_single > tj) {
                    best_j_pos_single = j;
                    best_t_pos_single = tj;
                }
            } else if (u_dot_pj < u_dot_p0) {
                if (best_j_neg_single < 0 || best_t_neg_single < tj) {
                    best_j_neg_single = j;
                    best_t_neg_single = tj;
                }
            }
        }
    }

    __local int best_j_pos_local[INTERSECT_RAY_INDEXED_GS0][INTERSECT_RAY_INDEXED_GS1];
    __local float best_t_pos_local[INTERSECT_RAY_INDEXED_GS0][INTERSECT_RAY_INDEXED_GS1];
    __local int best_j_neg_local[INTERSECT_RAY_INDEXED_GS0][INTERSECT_RAY_INDEXED_GS1];
    __local float best_t_neg_local[INTERSECT_RAY_INDEXED_GS0][INTERSECT_RAY_INDEXED_GS1];
    best_j_pos_local[loc_0][loc_1] = best_j_pos_single;
    best_t_pos_local[loc_0][loc_1] = best_t_pos_single;
    best_j_neg_local[loc_0][loc_1] = best_j_neg_single;
    best_t_neg_local[loc_0][loc_1] = best_t_neg_single;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (ref_i < ref_n && loc_0 == 0) {
        best_j_pos_single = -1;
        best_t_pos_single = FLT_MAX;
        best_j_neg_single = -1;
        best_t_neg_single = -FLT_MAX;
        for (int i = 0; i < INTERSECT_RAY_INDEXED_GS0; i++) {
            if (best_j_pos_single < 0 || best_t_pos_single > best_t_pos_local[i][loc_1]) {
                best_j_pos_single = best_j_pos_local[i][loc_1];
                best_t_pos_single = best_t_pos_local[i][loc_1];
            }
            if (best_j_neg_single < 0 || best_t_neg_single < best_t_neg_local[i][loc_1]) {
                best_j_neg_single = best_j_neg_local[i][loc_1];
                best_t_neg_single = best_t_neg_local[i][loc_1];
            }
        }
        best_j_pos[ref_i] = best_j_pos_single;
        best_j_neg[ref_i] = best_j_neg_single;
    }
}


#ifndef INTERSECT_RAY_INDEXED_BOUNDARY_GS0
#define INTERSECT_RAY_INDEXED_BOUNDARY_GS0 1
#endif
#ifndef INTERSECT_RAY_INDEXED_BOUNDARY_GS1
#define INTERSECT_RAY_INDEXED_BOUNDARY_GS1 1
#endif
__kernel void intersect_ray_indexed_boundary(__global const float *data_squared,
                                             __global const float *ref_dot_data,
                                             __global const float *u_dot_data,
                                             __global const int   *j0,  // sorted locally, split with -1
                                             __global const int   *j0_indices,
                                             __global const int   *u_indices,
                                             __global       int   *best_j_pos,
                                             __global       int   *best_j_neg,
                                             int data_n, int ref_n, int u_n) {
    // ti = (p0^2 - pj^2 + 2ref_dot_pj - 2ref_dot_p0) / (2u_dot_p0 - 2u_dot_pj)
    const uint ref_i = get_global_id(1);

    const uint loc_0 = get_local_id(0); // data_n
    const uint loc_1 = get_local_id(1); // ref_n

    int best_j_pos_single = -1;
    float best_t_pos_single = FLT_MAX;
    int best_j_neg_single = -1;
    float best_t_neg_single = -FLT_MAX;

    int u_i = -1;
    int j0_current = -1;
    int j0_i = -1;
    float p0_squared = 0.0f;
    float ref_dot_p0 = 0.0f;
    float u_dot_p0 = 0.0f;

    if (ref_i < ref_n) {
        u_i = u_indices[ref_i];
        j0_i = j0_indices[ref_i];
        j0_current = j0[j0_i];
        p0_squared = data_squared[j0_current];
        ref_dot_p0 = ref_dot_data[ref_i * data_n + j0_current];
        u_dot_p0 = u_dot_data[u_i * data_n + j0_current];
    }

    if (ref_i < ref_n) {
        for (int j = loc_0; j < data_n; j += INTERSECT_RAY_INDEXED_BOUNDARY_GS0) {
            while (j0_current >= 0 && j0_current < j) {
                j0_current = j0[++j0_i];
            }
            if (j == j0_current) {
                continue;
            }
            float pj_squared = data_squared[j];
            float ref_dot_pj = ref_dot_data[ref_i * data_n + j];
            float u_dot_pj = u_dot_data[u_i * data_n + j];

            float tj = (p0_squared - pj_squared - 2 * (ref_dot_p0 - ref_dot_pj)) / (2 * (u_dot_p0 - u_dot_pj));
            if (isnan(tj) || isinf(tj)) {
                continue;
            }

            if (u_dot_pj > u_dot_p0) {
                if (best_j_pos_single < 0 || best_t_pos_single > tj) {
                    best_j_pos_single = j;
                    best_t_pos_single = tj;
                }
            } else if (u_dot_pj < u_dot_p0) {
                if (best_j_neg_single < 0 || best_t_neg_single < tj) {
                    best_j_neg_single = j;
                    best_t_neg_single = tj;
                }
            }
        }
    }

    __local int best_j_pos_local[INTERSECT_RAY_INDEXED_BOUNDARY_GS0][INTERSECT_RAY_INDEXED_BOUNDARY_GS1];
    __local float best_t_pos_local[INTERSECT_RAY_INDEXED_BOUNDARY_GS0][INTERSECT_RAY_INDEXED_BOUNDARY_GS1];
    __local int best_j_neg_local[INTERSECT_RAY_INDEXED_BOUNDARY_GS0][INTERSECT_RAY_INDEXED_BOUNDARY_GS1];
    __local float best_t_neg_local[INTERSECT_RAY_INDEXED_BOUNDARY_GS0][INTERSECT_RAY_INDEXED_BOUNDARY_GS1];
    best_j_pos_local[loc_0][loc_1] = best_j_pos_single;
    best_t_pos_local[loc_0][loc_1] = best_t_pos_single;
    best_j_neg_local[loc_0][loc_1] = best_j_neg_single;
    best_t_neg_local[loc_0][loc_1] = best_t_neg_single;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (ref_i < ref_n && loc_0 == 0) {
        best_j_pos_single = -1;
        best_t_pos_single = FLT_MAX;
        best_j_neg_single = -1;
        best_t_neg_single = -FLT_MAX;
        for (int i = 0; i < INTERSECT_RAY_INDEXED_BOUNDARY_GS0; i++) {
            if (best_j_pos_single < 0 || best_t_pos_single > best_t_pos_local[i][loc_1]) {
                best_j_pos_single = best_j_pos_local[i][loc_1];
                best_t_pos_single = best_t_pos_local[i][loc_1];
            }
            if (best_j_neg_single < 0 || best_t_neg_single < best_t_neg_local[i][loc_1]) {
                best_j_neg_single = best_j_neg_local[i][loc_1];
                best_t_neg_single = best_t_neg_local[i][loc_1];
            }
        }
        best_j_pos[ref_i] = best_j_pos_single;
        best_j_neg[ref_i] = best_j_neg_single;
    }
}

__kernel void move_ref_points(__global       float *ref_dot_data,
                              __global const float *u_dot_data,
                              __global const int   *u_indices,
                              __global const float *t_picked,
                              int data_n, int ref_n, int u_n) {
    const uint data_i = get_global_id(0);
    const uint ref_i = get_global_id(1);

    if (ref_i < ref_n && data_i < data_n) {
        float u_dot_pj = u_dot_data[u_indices[ref_i] * data_n + data_i];
        ref_dot_data[ref_i * data_n + data_i] = ref_dot_data[ref_i * data_n + data_i] + u_dot_pj * t_picked[ref_i];
    }
}
