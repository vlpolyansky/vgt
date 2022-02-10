#include "kernels_gpu.h"
#include "../cl/voronoi_cl.h"

size_t get_workgroup_size(const ocl::Kernel &kernel) {
    return kernel.get_kernel(context->cl())->workGroupSize();
}

vec<int> optimal_update_data_squared_gs(
        size_t max_total, const vec<size_t> &max_local, int data_n, int dim) {
    vec<int> gs = {1, 1};
    size_t total = 1;

    while (gs[0] < max_local[0] && total < max_total
           && gs[0] * gs[0] * 4 <= dim) {
        gs[0] <<= 1;
        total <<= 1;
    }
    while (gs[1] < max_local[1] &&
           total < max_total) {
        gs[1] <<= 1;
        total <<= 1;
    }

    return gs;
}

vec<int> optimal_matmul_gs(
        size_t max_total, const vec<size_t> &max_local, int data_n, int dim) {
    vec<int> gs = {1, 1, 1};
    size_t total = 1;

    while (gs[0] < max_local[0] && total < max_total
           && gs[0] * gs[0] * 4 <= dim) {
        gs[0] <<= 1;
        total <<= 1;
    }
    while (gs[1] < max_local[1] && gs[2] < max_local[2] &&
           total * 2 < max_total) {
        gs[1] <<= 1;
        gs[2] <<= 1;
        total <<= 2;
    }

    return gs;
}

vec<int> optimal_intersect_ray_bruteforce_gs(
        size_t max_total, const vec<size_t> &max_local, int data_n, int dim) {
    vec<int> gs = {1, 1, 1};
    size_t total = 1;

    while (gs[0] < max_local[0] && total < max_total
           && gs[0] * gs[0] * 4 <= dim) {
        gs[0] <<= 1;
        total <<= 1;
    }
    while (gs[1] < max_local[1] && gs[2] < max_local[2] &&
           total * 2 < max_total) {
        gs[1] <<= 1;
        gs[2] <<= 1;
        total <<= 2;
    }

    return gs;
}

vec<int> optimal_intersect_ray_indexed_gs(
        size_t max_total, const vec<size_t> &max_local, int data_n, int dim) {
    vec<int> gs = {1, 1};
    size_t total = 1;

    while (gs[0] < max_local[0] && total < max_total
           && gs[0] * gs[0] * 4 <= dim) {
        gs[0] <<= 1;
        total <<= 1;
    }
    while (gs[1] < max_local[1] &&
           total < max_total) {
        gs[1] <<= 1;
        total <<= 1;
    }

    return gs;
}

vec<int> optimal_move_ref_points_gs(
        size_t max_total, const vec<size_t> &max_local, int data_n, int dim) {
    vec<int> gs = {1, 1};
    size_t total = 1;

    while (gs[0] < max_local[0] && total < max_total && gs[0] < data_n) {
        gs[0] <<= 1;
        total <<= 1;
    }
    while (gs[1] < max_local[1] && total < max_total) {
        gs[1] <<= 1;
        total <<= 1;
    }

    return gs;
}

void EuclideanKernelGPU::init() {
    // initialize gpu
    if (!context) {
        context = std::make_shared<gpu::Context>();
        std::vector<gpu::Device> devices = gpu::enumDevices();
        if (devices.empty()) {
            throw std::runtime_error("No OpenCL devices found!");
        }
        std::cout << "Available devices:" << std::endl;
        for (auto &d : devices) {
            std::cout << "    " << d.name << std::endl;
        }
        auto picked_device = devices[0];  // todo: make selection
        std::cout << "Using the device " << picked_device.name << std::endl;

        context->init(picked_device.device_id_opencl);
        context->activate();

        device = std::make_shared<gpu::Device>(picked_device);
    }

    data_n = points.cols();
    dim = points.rows();

    std::vector<size_t> max_local = context->getMaxWorkItemSizes();

    // preliminary compile kernels
    update_data_squared_kernel = init_kernel("update_data_squared",
                                             "-D UPDATE_DATA_SQUARED_GS0=" + std::to_string(update_data_squared_gs[0])
                                             + " -D UPDATE_DATA_SQUARED_GS1=" + std::to_string(update_data_squared_gs[1])
                                             + " -D UPDATE_DATA_SQUARED_GS2=" + std::to_string(update_data_squared_gs[2]));
    matmul_kernel = init_kernel("matmul",
                                "-D MATMUL_GS0=" + std::to_string(matmul_gs[0])
                                + " -D MATMUL_GS1=" + std::to_string(matmul_gs[1])
                                + " -D MATMUL_GS2=" + std::to_string(matmul_gs[2]));
    intersect_ray_bruteforce_kernel = init_kernel("intersect_ray_bruteforce",
                                                  "-D INTERSECT_RAY_BRUTEFORCE_GS0=" + std::to_string(intersect_ray_bruteforce_gs[0])
                                                  + " -D INTERSECT_RAY_BRUTEFORCE_GS1=" + std::to_string(intersect_ray_bruteforce_gs[1])
                                                  + " -D INTERSECT_RAY_BRUTEFORCE_GS2=" + std::to_string(intersect_ray_bruteforce_gs[2]));
    intersect_ray_indexed_kernel = init_kernel("intersect_ray_indexed",
                                               "-D INTERSECT_RAY_INDEXED_GS0=" + std::to_string(intersect_ray_indexed_gs[0])
                                               + " -D INTERSECT_RAY_INDEXED_GS1=" + std::to_string(intersect_ray_indexed_gs[1]));
    intersect_ray_indexed_boundary_kernel = init_kernel("intersect_ray_indexed_boundary",
                                                        "-D INTERSECT_RAY_INDEXED_BOUNDARY_GS0=" + std::to_string(intersect_ray_indexed_boundary_gs[0])
                                                        + " -D INTERSECT_RAY_INDEXED_BOUNDARY_GS1=" + std::to_string(intersect_ray_indexed_boundary_gs[1]));
    move_ref_points_kernel = init_kernel("move_ref_points");


    // compute optimal workgroup sizes
    update_data_squared_gs = optimal_update_data_squared_gs(
            get_workgroup_size(update_data_squared_kernel), max_local, data_n, dim); // {32, 4}
    matmul_gs = optimal_matmul_gs(
            get_workgroup_size(matmul_kernel), max_local, data_n, dim); // {32, 2, 2}
    intersect_ray_bruteforce_gs = optimal_intersect_ray_bruteforce_gs(
            get_workgroup_size(intersect_ray_bruteforce_kernel), max_local, data_n, dim); // {32, 2, 2};
    intersect_ray_indexed_gs = optimal_intersect_ray_indexed_gs(
            get_workgroup_size(intersect_ray_indexed_kernel), max_local, data_n, dim); // {32, 4};
    intersect_ray_indexed_boundary_gs = optimal_intersect_ray_indexed_gs( // can reuse here
            get_workgroup_size(intersect_ray_indexed_boundary_kernel), max_local, data_n, dim); // {32, 4};
    move_ref_points_gs = optimal_move_ref_points_gs(
            get_workgroup_size(move_ref_points_kernel), max_local, data_n, dim); // {16, 16};

    // compile kernels
    update_data_squared_kernel = init_kernel("update_data_squared",
                                             "-D UPDATE_DATA_SQUARED_GS0=" + std::to_string(update_data_squared_gs[0])
                                             + " -D UPDATE_DATA_SQUARED_GS1=" + std::to_string(update_data_squared_gs[1])
                                             + " -D UPDATE_DATA_SQUARED_GS2=" + std::to_string(update_data_squared_gs[2]));
    matmul_kernel = init_kernel("matmul",
                                "-D MATMUL_GS0=" + std::to_string(matmul_gs[0])
                                + " -D MATMUL_GS1=" + std::to_string(matmul_gs[1])
                                + " -D MATMUL_GS2=" + std::to_string(matmul_gs[2]));
    intersect_ray_bruteforce_kernel = init_kernel("intersect_ray_bruteforce",
                                                  "-D INTERSECT_RAY_BRUTEFORCE_GS0=" + std::to_string(intersect_ray_bruteforce_gs[0])
                                                  + " -D INTERSECT_RAY_BRUTEFORCE_GS1=" + std::to_string(intersect_ray_bruteforce_gs[1])
                                                  + " -D INTERSECT_RAY_BRUTEFORCE_GS2=" + std::to_string(intersect_ray_bruteforce_gs[2]));
    intersect_ray_indexed_kernel = init_kernel("intersect_ray_indexed",
                                               "-D INTERSECT_RAY_INDEXED_GS0=" + std::to_string(intersect_ray_indexed_gs[0])
                                               + " -D INTERSECT_RAY_INDEXED_GS1=" + std::to_string(intersect_ray_indexed_gs[1]));
    intersect_ray_indexed_boundary_kernel = init_kernel("intersect_ray_indexed_boundary",
                                                        "-D INTERSECT_RAY_INDEXED_BOUNDARY_GS0=" + std::to_string(intersect_ray_indexed_boundary_gs[0])
                                                        + " -D INTERSECT_RAY_INDEXED_BOUNDARY_GS1=" + std::to_string(intersect_ray_indexed_boundary_gs[1]));
    move_ref_points_kernel = init_kernel("move_ref_points");

    // read data
    writeN(data_gpu, points);

    data_squared_gpu.resizeN(data_n);
    update_data_squared_kernel.exec(work_size(update_data_squared_gs[0], update_data_squared_gs[1],
                                              update_data_squared_gs[0], data_n),
                                    data_gpu, data_squared_gpu,
                                    data_n, dim);
}

EuclideanKernelGPU::EuclideanKernelGPU(const dmatrix &points, const ptr<Bounds> &bounds) : EuclideanKernel(points, bounds) {
    init();
}

int EuclideanKernelGPU::estimate_max_block_size(int planned_u_n) const {
    auto available_memory = device->getFreeMemory();
    long long int max_alloc = context->getMaxMemAlloc() / (data_n * 4);
    long long int tmp;
    std::cout << "Memory available (VRAM, bytes): " << available_memory << std::endl;
    if (planned_u_n < 0) {  // assume u_n = ref_n
        double a = 3;
        double b = 3 + 2 * dim + 2 * data_n;
        double c = double(data_n) * (dim + 1) - available_memory * 0.25;
        tmp = static_cast<int>((-b + math::sqrt(b * b - 4 * a * c)) / (2 * a) * 0.9);
    } else {
        double b = dim + data_n + 3 * planned_u_n + 3;
        double c = double(data_n) * dim + data_n + double(planned_u_n) * (data_n + dim) - available_memory * 0.25;
        tmp = static_cast<int>(-c / b * 0.9f);
    }
    tmp = math::min(tmp, max_alloc);
    return tmp < 1000000000 ? static_cast<int>(tmp) : -1;
}

void EuclideanKernelGPU::reset_reference_points_gpu(const dmatrix &ref_mat, const vec<int> &j0_list) {
    ref_n = ref_mat.cols();

    boundary_references = false;
    writeN(ref_gpu, ref_mat);

    ref_squared_gpu.resizeN(ref_n);
    update_data_squared_kernel.exec(work_size(update_data_squared_gs[0], update_data_squared_gs[1],
                                              update_data_squared_gs[0], ref_n),
                                    ref_gpu, ref_squared_gpu,
                                    ref_n, dim);

    j0_gpu.resizeN(j0_list.size());
    j0_gpu.writeN(j0_list.data(), j0_list.size());


    ref_dot_data_gpu.resizeN(ref_n * data_n);
    matmul_kernel.exec(work_size(matmul_gs[0], matmul_gs[1], matmul_gs[2],
                                 matmul_gs[0], ref_n, data_n),
                       ref_gpu, data_gpu, ref_dot_data_gpu,
                       ref_n, data_n, dim);
}

void EuclideanKernelGPU::reset_reference_points_gpu(const dmatrix &ref_mat, const vec<IndexSet> &duals) {
    ref_n = ref_mat.cols();

    boundary_references = true;
    writeN(ref_gpu, ref_mat);

    vec<int> j0_list;
    vec<int> j0_indices;
    for (int i = 0; i < duals.size(); i++) {
        j0_indices.push_back(j0_list.size());
        vec<int> tmp = duals[i];
//        std::sort(tmp.begin(), tmp.end());
        for (int j = 0; j < tmp.size(); j++) {
            j0_list.push_back(tmp[j]);
        }
        j0_list.push_back(-1);
    }

    j0_gpu.resizeN(j0_list.size());
    j0_gpu.writeN(j0_list.data(), j0_list.size());
    j0_indices_gpu.resizeN(j0_indices.size());
    j0_indices_gpu.writeN(j0_indices.data(), j0_indices.size());

    ref_dot_data_gpu.resizeN(ref_n * data_n);
    matmul_kernel.exec(work_size(matmul_gs[0], matmul_gs[1], matmul_gs[2],
                                 matmul_gs[0], ref_n, data_n),
                       ref_gpu, data_gpu, ref_dot_data_gpu,
                       ref_n, data_n, dim);

    if (u_n >= 0) {
        u_dot_ref_gpu.resizeN(u_n * ref_n);
        matmul_kernel.exec(work_size(matmul_gs[0], matmul_gs[1], matmul_gs[2],
                                     matmul_gs[0], u_n, ref_n),
                           u_gpu, ref_gpu, u_dot_ref_gpu,
                           u_n, ref_n, dim);

    }
}

void EuclideanKernelGPU::reset_rays_gpu(const dmatrix &u_mat) {
    u_n = u_mat.cols();

    writeN(u_gpu, u_mat);

    u_dot_data_gpu.resizeN(u_n * data_n);
//    reset_rays_kernel.exec(work_size(0, 0),
    matmul_kernel.exec(work_size(matmul_gs[0], matmul_gs[1], matmul_gs[2],
                                 matmul_gs[0], u_n, data_n),
                       u_gpu, data_gpu, u_dot_data_gpu,
                       u_n, data_n, dim);
    if (ref_n >= 0) {
        u_dot_ref_gpu.resizeN(u_n * ref_n);
        matmul_kernel.exec(work_size(matmul_gs[0], matmul_gs[1], matmul_gs[2],
                                     matmul_gs[0], u_n, ref_n),
                           u_gpu, ref_gpu, u_dot_ref_gpu,
                           u_n, ref_n, dim);

    }
}

void EuclideanKernelGPU::intersect_ray_bruteforce_gpu(vec<int> *best_j_pos, vec<int> *best_j_neg) {
    ensure(best_j_pos->size() >= ref_n * u_n, "best_j_pos is too small");
    ensure(best_j_neg->size() >= ref_n * u_n, "best_j_neg is too small");

    best_j_pos_gpu.resizeN(ref_n * u_n);
    best_j_neg_gpu.resizeN(ref_n * u_n);
    vec<int> tmp(ref_n * u_n, -2);
    best_j_pos_gpu.writeN(tmp.data(), ref_n * u_n);
    intersect_ray_bruteforce_kernel.exec(work_size(intersect_ray_bruteforce_gs[0], intersect_ray_bruteforce_gs[1],
                                                   intersect_ray_bruteforce_gs[2], intersect_ray_bruteforce_gs[0], ref_n, u_n),
                                         data_squared_gpu, ref_squared_gpu,
                                         ref_dot_data_gpu, u_dot_data_gpu, u_dot_ref_gpu,
                                         j0_gpu,
                                         best_j_pos_gpu, best_j_neg_gpu,
                                         data_n, ref_n, u_n);

    best_j_pos_gpu.readN(best_j_pos->data(), ref_n * u_n);
    best_j_neg_gpu.readN(best_j_neg->data(), ref_n * u_n);
}

void
EuclideanKernelGPU::intersect_ray_indexed_gpu(const vec<int> &u_indices, vec<int> *best_j_pos, vec<int> *best_j_neg) {
    ensure(best_j_pos->size() >= ref_n, "best_j_pos is too small");
    ensure(best_j_neg->size() >= ref_n, "best_j_neg is too small");

    best_j_pos_gpu.resizeN(ref_n);
    best_j_neg_gpu.resizeN(ref_n);

    vec<int> tmp(ref_n, -2);
    best_j_pos_gpu.writeN(tmp.data(), ref_n);
    best_j_neg_gpu.writeN(tmp.data(), ref_n);

    u_indices_gpu.resizeN(ref_n);
    u_indices_gpu.writeN(u_indices.data(), ref_n);

    if (boundary_references) {
        intersect_ray_indexed_boundary_kernel.exec(work_size(intersect_ray_indexed_gs[0], intersect_ray_indexed_gs[1],
                                                             intersect_ray_indexed_gs[0], ref_n),
                                                   data_squared_gpu, ref_dot_data_gpu, u_dot_data_gpu,
                                                   j0_gpu, j0_indices_gpu, u_indices_gpu,
                                                   best_j_pos_gpu, best_j_neg_gpu,
                                                   data_n, ref_n, u_n);
//        std::cout << "HERE\n" << std::endl;
    } else {
        intersect_ray_indexed_kernel.exec(work_size(intersect_ray_indexed_gs[0], intersect_ray_indexed_gs[1],
                                                    intersect_ray_indexed_gs[0], ref_n),
                                          data_squared_gpu, ref_dot_data_gpu, u_dot_data_gpu,
                                          j0_gpu, u_indices_gpu,
                                          best_j_pos_gpu, best_j_neg_gpu,
                                          data_n, ref_n, u_n);
    }

    best_j_pos_gpu.readN(best_j_pos->data(), ref_n);
    best_j_neg_gpu.readN(best_j_neg->data(), ref_n);
}

void EuclideanKernelGPU::move_reference_points(const vec<ftype> &t) {
    // ref[i] <- ref[i] + u[u_indices[i]] * t[i]
    // ref_dot_data(i, j) <- ref_dot_data(i, j) + u_dot_data(u_indices[i], j) * t[i]
    // u_indices should already be written
    writeN(t_picked_gpu, t);
    move_ref_points_kernel.exec(work_size(move_ref_points_gs[0], move_ref_points_gs[1], data_n, ref_n),
                                ref_dot_data_gpu, u_dot_data_gpu, u_indices_gpu, t_picked_gpu,
                                data_n, ref_n, u_n);

}

void EuclideanKernelGPU::update_inserted_points(int n_new_points) {
    EuclideanKernel::update_inserted_points(n_new_points);
    init();
}


ocl::Kernel init_kernel(const std::string &name, const std::string &defines) {
    return ocl::Kernel(voronoi_kernel_sources, voronoi_kernel_sources_length, name, defines);
}

gpu::WorkSize work_size(unsigned int gsX, unsigned int wsX) {
    return {gsX, (wsX + gsX - 1) / gsX * gsX};
}

gpu::WorkSize work_size(unsigned int gsX, unsigned int gsY, unsigned int wsX, unsigned int wsY) {
    return {gsX, gsY, (wsX + gsX - 1) / gsX * gsX, (wsY + gsY - 1) / gsY * gsY};
}

gpu::WorkSize
work_size(unsigned int gsX, unsigned int gsY, unsigned int gsZ, unsigned int wsX, unsigned int wsY, unsigned int wsZ) {
    return {gsX, gsY, gsZ,
            (wsX + gsX - 1) / gsX * gsX, (wsY + gsY - 1) / gsY * gsY, (wsZ + gsZ - 1) / gsZ * gsZ};
}

void writeN(gpu::gpu_mem_32f &buffer, const dmatrix &data) {
    vec<float> tmp(data.size());
    std::transform(data.data(), data.data() + data.size(), tmp.data(), [](ftype a) -> float {return float(a);});
    buffer.resizeN(tmp.size());
    buffer.writeN(tmp.data(), tmp.size());
}

void writeN(gpu::gpu_mem_32f &buffer, const vec<ftype> &data) {
    vec<float> tmp(data.size());
    std::transform(data.data(), data.data() + data.size(), tmp.data(), [](ftype a) -> float {return float(a);});
    buffer.resizeN(tmp.size());
    buffer.writeN(tmp.data(), tmp.size());
}
