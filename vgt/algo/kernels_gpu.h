#pragma once

#include <libgpu/device.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/misc.h>

#include "kernels.h"

static ptr<gpu::Context> context;
static ptr<gpu::Device> device;

class EuclideanKernelGPU : public EuclideanKernel {
private:
    void init();
public:
    explicit EuclideanKernelGPU(const dmatrix &points, const ptr<Bounds> &bounds = std::make_shared<Unbounded>());

    void reset_reference_points_gpu(const dmatrix &ref_mat, const vec<int> &j0_list);

    void reset_reference_points_gpu(const dmatrix &ref_mat, const vec<IndexSet> &duals);

    void reset_rays_gpu(const dmatrix &u_mat);

    void intersect_ray_bruteforce_gpu(/* [ref][u] */vec<int> *best_j_pos, vec<int> *best_j_neg);

    void intersect_ray_indexed_gpu(const vec<int> &u_indices, vec<int> *best_j_pos, vec<int> *best_j_neg);

    void move_reference_points(const vec<ftype> &t);

    int estimate_max_block_size(int planned_u_n = -1) const;

    void update_inserted_points(int n_new_points) override;

protected:
    int data_n = -1;
    int dim = -1;
    int ref_n = -1;
    int u_n = -1;

    bool boundary_references = false;

    gpu::gpu_mem_32f data_gpu;  // n * d
    gpu::gpu_mem_32f data_squared_gpu; // n
    gpu::gpu_mem_32f ref_gpu;   // ref_n * d
    gpu::gpu_mem_32f ref_squared_gpu;   // ref_n
    gpu::gpu_mem_32i j0_gpu;   // ref_n
    gpu::gpu_mem_32i j0_indices_gpu;   // ref_n (only for boundary points)
    gpu::gpu_mem_32f ref_dot_data_gpu;  // ref_n * data_n
    gpu::gpu_mem_32f u_gpu;   // u_n * d
    gpu::gpu_mem_32f u_dot_data_gpu;  // u_n * data_n
    gpu::gpu_mem_32f u_dot_ref_gpu;  // u_n * ref_n
    gpu::gpu_mem_32i best_j_pos_gpu;  // ref_n * u_n
    gpu::gpu_mem_32i best_j_neg_gpu;  // ref_n * u_n
    gpu::gpu_mem_32i u_indices_gpu;  //
    gpu::gpu_mem_32f t_picked_gpu;

    vec<int> update_data_squared_gs = {32, 4};
    vec<int> matmul_gs = {32, 2, 2};
    vec<int> intersect_ray_bruteforce_gs = {32, 2, 2};
    vec<int> intersect_ray_indexed_gs = {32, 4};
    vec<int> intersect_ray_indexed_boundary_gs = {32, 4};
    vec<int> move_ref_points_gs = {16, 16};

    ocl::Kernel update_data_squared_kernel;
    ocl::Kernel matmul_kernel;
    ocl::Kernel intersect_ray_bruteforce_kernel;
    ocl::Kernel intersect_ray_indexed_kernel;
    ocl::Kernel intersect_ray_indexed_boundary_kernel;
    ocl::Kernel move_ref_points_kernel;
//    ocl::Kernel reset_reference_points_kernel;
//    ocl::Kernel reset_rays_kernel;
};


// -- OpenCL --
ocl::Kernel init_kernel(const std::string &name, const std::string &defines = "");
gpu::WorkSize work_size(unsigned int gsX, unsigned int wsX);
gpu::WorkSize work_size(unsigned int gsX, unsigned int gsY, unsigned int wsX, unsigned int wsY);
gpu::WorkSize work_size(unsigned int gsX, unsigned int gsY, unsigned int gsZ,
                        unsigned int wsX, unsigned int wsY, unsigned int wsZ);
void writeN(gpu::gpu_mem_32f &buffer, const dmatrix &data);
void writeN(gpu::gpu_mem_32f &buffer, const vec<ftype> &data);