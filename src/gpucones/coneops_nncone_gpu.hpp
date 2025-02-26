#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/fill.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>
#include <thrust/transform_reduce.h>
#include <thrust/transform_scan.h>
#include <thrust/sequence.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/zip_iterator.h>

template <typename T>
__global__ void _kernel_step_length_nonnegative(
    T* dz,
    T* ds,
    T* z,
    T* s,
    T* α,
    int len_rng,
    T αmax
) {
    int i = (blockIdx.x - 1) * blockDim.x + threadIdx.x;

    if (i <= len_rng) {
        T αz = dz[i] < 0 ? (min(αmax, -z[i] / dz[i])) : αmax;
        T αs = ds[i] < 0 ? (min(αmax, -s[i] / ds[i])) : αmax;
        α[i] = min(αz, αs);
    }
}

template <typename T>
T step_length_nonnegative(
    thrust::device_vector<T>& dz,
    thrust::device_vector<T>& ds,
    thrust::device_vector<T>& z,
    thrust::device_vector<T>& s,
    thrust::device_vector<T>& α,
    T αmax,
    thrust::device_vector<int>& rng_cones,
    thrust::device_vector<int>& idx_inq
) {
    for (int i : idx_inq) {
        int len_nn = rng_cones[i + 1] - rng_cones[i];
        int rng_cone_i = rng_cones[i];
        T* dzi = thrust::raw_pointer_cast(dz.data()) + rng_cone_i;
        T* dsi = thrust::raw_pointer_cast(ds.data()) + rng_cone_i;
        T* zi = thrust::raw_pointer_cast(z.data()) + rng_cone_i;
        T* si = thrust::raw_pointer_cast(s.data()) + rng_cone_i;
        T* αi = thrust::raw_pointer_cast(α.data()) + rng_cone_i;

        int threads = std::min(len_nn, 1024);
        int blocks = (len_nn + threads - 1) / threads;

        _kernel_step_length_nonnegative<<<blocks, threads>>>(
            dzi,
            dsi,
            zi,
            si,
            αi,
            len_nn,
            αmax
        );
        cudaDeviceSynchronize();

        αmax = thrust::reduce(α.begin() + rng_cone_i, α.begin() + rng_cone_i + len_nn, αmax, thrust::minimum<T>());
    }

    return αmax;
}

template <typename T>
__global__ void _kernel_compute_barrier_nonnegative(
    T* barrier,
    T* z,
    T* s,
    T* dz,
    T* ds,
    T α,
    int len_nn
) {
    int i = (blockIdx.x - 1) * blockDim.x + threadIdx.x;

    if (i <= len_nn) {
        barrier[i] = -log((s[i] + α * ds[i]) * (z[i] + α * dz[i]));
    }
}

template <typename T>
T compute_barrier_nonnegative(
    thrust::device_vector<T>& work,
    thrust::device_vector<T>& z,
    thrust::device_vector<T>& s,
    thrust::device_vector<T>& dz,
    thrust::device_vector<T>& ds,
    T α,
    thrust::device_vector<int>& rng_cones,
    thrust::device_vector<int>& idx_inq
) {
    T barrier = 0;

    for (int i : idx_inq) {
        int len_nn = rng_cones[i + 1] - rng_cones[i];
        int rng_cone_i = rng_cones[i];
        T* worki = thrust::raw_pointer_cast(work.data()) + rng_cone_i;
        T* zi = thrust::raw_pointer_cast(z.data()) + rng_cone_i;
        T* si = thrust::raw_pointer_cast(s.data()) + rng_cone_i;
        T* dzi = thrust::raw_pointer_cast(dz.data()) + rng_cone_i;
        T* dsi = thrust::raw_pointer_cast(ds.data()) + rng_cone_i;

        int threads = std::min(len_nn, 1024);
        int blocks = (len_nn + threads - 1) / threads;

        _kernel_compute_barrier_nonnegative<<<blocks, threads>>>(
            worki,
            zi,
            si,
            dzi,
            dsi,
            α,
            len_nn
        );
        cudaDeviceSynchronize();

        barrier += thrust::reduce(work.begin() + rng_cone_i, work.begin() + rng_cone_i + len_nn, static_cast<T>(0), thrust::plus<T>());
    }

    return barrier;
}

template <typename T>
void margins_nonnegative(
    thrust::device_vector<T>& z,
    thrust::device_vector<T>& α,
    thrust::device_vector<int>& rng_cones,
    thrust::device_vector<int>& idx_inq,
    T& αmin,
    T& margin
) {
    for (int i : idx_inq) {
        int rng_cone_i = rng_cones[i];
        T* zi = thrust::raw_pointer_cast(z.data()) + rng_cone_i;
        T* αi = thrust::raw_pointer_cast(α.data()) + rng_cone_i;

        αmin = std::min(αmin, thrust::reduce(zi, zi + (rng_cones[i + 1] - rng_cones[i]), αmin, thrust::minimum<T>()));
        thrust::transform(zi, zi + (rng_cones[i + 1] - rng_cones[i]), αi, thrust::placeholders::_1 = thrust::maximum(thrust::placeholders::_1, static_cast<T>(0)));
        margin += thrust::reduce(αi, αi + (rng_cones[i + 1] - rng_cones[i]), static_cast<T>(0), thrust::plus<T>());
    }
}

template <typename T>
void scaled_unit_shift_nonnegative(
    thrust::device_vector<T>& z,
    thrust::device_vector<int>& rng_cones,
    thrust::device_vector<int>& idx_inq,
    T α
) {
    for (int i : idx_inq) {
        int rng_cone_i = rng_cones[i];
        T* zi = thrust::raw_pointer_cast(z.data()) + rng_cone_i;

        thrust::transform(zi, zi + (rng_cones[i + 1] - rng_cones[i]), zi, thrust::placeholders::_1 + α);
    }
}

template <typename T>
void unit_initialization_nonnegative(
    thrust::device_vector<T>& z,
    thrust::device_vector<T>& s,
    thrust::device_vector<int>& rng_cones,
    thrust::device_vector<int>& idx_inq
) {
    for (int i : idx_inq) {
        int rng_cone_i = rng_cones[i];
        T* zi = thrust::raw_pointer_cast(z.data()) + rng_cone_i;
        T* si = thrust::raw_pointer_cast(s.data()) + rng_cone_i;

        thrust::fill(zi, zi + (rng_cones[i + 1] - rng_cones[i]), static_cast<T>(1));
        thrust::fill(si, si + (rng_cones[i + 1] - rng_cones[i]), static_cast<T>(1));
    }
}

template <typename T>
void set_identity_scaling_nonnegative(
    thrust::device_vector<T>& w,
    thrust::device_vector<int>& rng_cones,
    thrust::device_vector<int>& idx_inq
) {
    for (int i : idx_inq) {
        int rng_cone_i = rng_cones[i];
        T* wi = thrust::raw_pointer_cast(w.data()) + rng_cone_i;

        thrust::fill(wi, wi + (rng_cones[i + 1] - rng_cones[i]), static_cast<T>(1));
    }
}

template <typename T>
void update_scaling_nonnegative(
    thrust::device_vector<T>& s,
    thrust::device_vector<T>& z,
    thrust::device_vector<T>& w,
    thrust::device_vector<T>& λ,
    thrust::device_vector<int>& rng_cones,
    thrust::device_vector<int>& idx_inq
) {
    for (int i : idx_inq) {
        int rng_cone_i = rng_cones[i];
        T* si = thrust::raw_pointer_cast(s.data()) + rng_cone_i;
        T* zi = thrust::raw_pointer_cast(z.data()) + rng_cone_i;
        T* λi = thrust::raw_pointer_cast(λ.data()) + rng_cone_i;
        T* wi = thrust::raw_pointer_cast(w.data()) + rng_cone_i;

        thrust::transform(si, si + (rng_cones[i + 1] - rng_cones[i]), zi, λi, thrust::placeholders::_1 = sqrt(thrust::placeholders::_1 * thrust::placeholders::_2));
        thrust::transform(si, si + (rng_cones[i + 1] - rng_cones[i]), zi, wi, thrust::placeholders::_1 = sqrt(thrust::placeholders::_1 / thrust::placeholders::_2));
    }
}

template <typename T>
void get_Hs_nonnegative(
    thrust::device_vector<T>& Hsblocks,
    thrust::device_vector<T>& w,
    thrust::device_vector<int>& rng_cones,
    thrust::device_vector<int>& rng_blocks,
    thrust::device_vector<int>& idx_inq
) {
    for (int i : idx_inq) {
        int rng_cone_i = rng_cones[i];
        int rng_block_i = rng_blocks[i];
        T* wi = thrust::raw_pointer_cast(w.data()) + rng_cone_i;
        T* Hsblocki = thrust::raw_pointer_cast(Hsblocks.data()) + rng_block_i;

        thrust::transform(wi, wi + (rng_cones[i + 1] - rng_cones[i]), Hsblocki, thrust::placeholders::_1 = thrust::placeholders::_1 * thrust::placeholders::_1);
    }
}

template <typename T>
void mul_Hs_nonnegative(
    thrust::device_vector<T>& y,
    thrust::device_vector<T>& x,
    thrust::device_vector<T>& w,
    thrust::device_vector<int>& rng_cones,
    thrust::device_vector<int>& idx_inq
) {
    for (int i : idx_inq) {
        int rng_cone_i = rng_cones[i];
        T* wi = thrust::raw_pointer_cast(w.data()) + rng_cone_i;
        T* xi = thrust::raw_pointer_cast(x.data()) + rng_cone_i;
        T* yi = thrust::raw_pointer_cast(y.data()) + rng_cone_i;

        thrust::transform(wi, wi + (rng_cones[i + 1] - rng_cones[i]), xi, yi, thrust::placeholders::_1 = thrust::placeholders::_1 * thrust::placeholders::_1 * thrust::placeholders::_2);
    }
}

template <typename T>
void affine_ds_nonnegative(
    thrust::device_vector<T>& ds,
    thrust::device_vector<T>& λ,
    thrust::device_vector<int>& rng_cones,
    thrust::device_vector<int>& idx_inq
) {
    for (int i : idx_inq) {
        int rng_cone_i = rng_cones[i];
        T* dsi = thrust::raw_pointer_cast(ds.data()) + rng_cone_i;
        T* λi = thrust::raw_pointer_cast(λ.data()) + rng_cone_i;

        thrust::transform(λi, λi + (rng_cones[i + 1] - rng_cones[i]), dsi, thrust::placeholders::_1 = thrust::placeholders::_1 * thrust::placeholders::_1);
    }
}

template <typename T>
void combined_ds_shift_nonnegative(
    thrust::device_vector<T>& shift,
    thrust::device_vector<T>& step_z,
    thrust::device_vector<T>& step_s,
    thrust::device_vector<T>& w,
    T σμ,
    thrust::device_vector<int>& rng_cones,
    thrust::device_vector<int>& idx_inq
) {
    for (int i : idx_inq) {
        int rng_cone_i = rng_cones[i];
        T* shift_i = thrust::raw_pointer_cast(shift.data()) + rng_cone_i;
        T* step_zi = thrust::raw_pointer_cast(step_z.data()) + rng_cone_i;
        T* step_si = thrust::raw_pointer_cast(step_s.data()) + rng_cone_i;
        T* wi = thrust::raw_pointer_cast(w.data()) + rng_cone_i;

        thrust::device_vector<T> tmp(step_zi, step_zi + (rng_cones[i + 1] - rng_cones[i]));

        thrust::transform(tmp.begin(), tmp.end(), wi, step_zi, thrust::placeholders::_1 = thrust::placeholders::_1 * thrust::placeholders::_2);
        thrust::transform(tmp.begin(), tmp.end(), wi, step_si, thrust::placeholders::_1 = thrust::placeholders::_1 / thrust::placeholders::_2);
        thrust::transform(step_si, step_si + (rng_cones[i + 1] - rng_cones[i]), step_zi, shift_i, thrust::placeholders::_1 = thrust::placeholders::_1 * thrust::placeholders::_2 - σμ);
    }
}

template <typename T>
void Δs_from_Δz_offset_nonnegative(
    thrust::device_vector<T>& out,
    thrust::device_vector<T>& ds,
    thrust::device_vector<T>& z,
    thrust::device_vector<int>& rng_cones,
    thrust::device_vector<int>& idx_inq
) {
    for (int i : idx_inq) {
        int rng_cone_i = rng_cones[i];
        T* out_i = thrust::raw_pointer_cast(out.data()) + rng_cone_i;
        T* dsi = thrust::raw_pointer_cast(ds.data()) + rng_cone_i;
        T* zi = thrust::raw_pointer_cast(z.data()) + rng_cone_i;

        thrust::transform(dsi, dsi + (rng_cones[i + 1] - rng_cones[i]), zi, out_i, thrust::placeholders::_1 = thrust::placeholders::_1 / thrust::placeholders::_2);
    }
}
