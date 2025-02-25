#include <vector>
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
__global__ void _kernel_scaled_unit_shift_zero(
    T* z,
    int* rng_cones,
    int* idx_eq,
    int n_eq
) {
    int i = (blockIdx.x - 1) * blockDim.x + threadIdx.x;

    if (i <= n_eq) {
        int rng_cone_i = rng_cones[idx_eq[i]];
        z[rng_cone_i] = static_cast<T>(0);
    }
}

template <typename T>
void scaled_unit_shift_zero(
    thrust::device_vector<T>& z,
    thrust::device_vector<int>& rng_cones,
    thrust::device_vector<int>& idx_eq
) {
    int n_eq = idx_eq.size();
    int threads = std::min(n_eq, 1024);
    int blocks = (n_eq + threads - 1) / threads;

    _kernel_scaled_unit_shift_zero<<<blocks, threads>>>(
        thrust::raw_pointer_cast(z.data()),
        thrust::raw_pointer_cast(rng_cones.data()),
        thrust::raw_pointer_cast(idx_eq.data()),
        n_eq
    );
    cudaDeviceSynchronize();
}

template <typename T>
__global__ void _kernel_unit_initialization_zero(
    T* z,
    T* s,
    int* rng_cones,
    int* idx_eq,
    int n_eq
) {
    int i = (blockIdx.x - 1) * blockDim.x + threadIdx.x;

    if (i <= n_eq) {
        int rng_cone_i = rng_cones[idx_eq[i]];
        z[rng_cone_i] = static_cast<T>(0);
        s[rng_cone_i] = static_cast<T>(0);
    }
}

template <typename T>
void unit_initialization_zero(
    thrust::device_vector<T>& z,
    thrust::device_vector<T>& s,
    thrust::device_vector<int>& rng_cones,
    thrust::device_vector<int>& idx_eq
) {
    int n_eq = idx_eq.size();
    int threads = std::min(n_eq, 1024);
    int blocks = (n_eq + threads - 1) / threads;

    _kernel_unit_initialization_zero<<<blocks, threads>>>(
        thrust::raw_pointer_cast(z.data()),
        thrust::raw_pointer_cast(s.data()),
        thrust::raw_pointer_cast(rng_cones.data()),
        thrust::raw_pointer_cast(idx_eq.data()),
        n_eq
    );
    cudaDeviceSynchronize();
}

template <typename T>
__global__ void _kernel_get_Hs_zero(
    T* Hsblocks,
    int* rng_blocks,
    int* idx_eq,
    int n_eq
) {
    int i = (blockIdx.x - 1) * blockDim.x + threadIdx.x;

    if (i <= n_eq) {
        Hsblocks[rng_blocks[idx_eq[i]]] = static_cast<T>(0);
    }
}

template <typename T>
void get_Hs_zero(
    thrust::device_vector<T>& Hsblocks,
    thrust::device_vector<int>& rng_blocks,
    thrust::device_vector<int>& idx_eq
) {
    int n_eq = idx_eq.size();
    int threads = std::min(n_eq, 1024);
    int blocks = (n_eq + threads - 1) / threads;

    _kernel_get_Hs_zero<<<blocks, threads>>>(
        thrust::raw_pointer_cast(Hsblocks.data()),
        thrust::raw_pointer_cast(rng_blocks.data()),
        thrust::raw_pointer_cast(idx_eq.data()),
        n_eq
    );
    cudaDeviceSynchronize();
}

template <typename T>
__global__ void _kernel_mul_Hs_zero(
    T* y,
    int* rng_cones,
    int* idx_eq,
    int n_eq
) {
    int i = (blockIdx.x - 1) * blockDim.x + threadIdx.x;

    if (i <= n_eq) {
        y[rng_cones[idx_eq[i]]] = static_cast<T>(0);
    }
}

template <typename T>
void mul_Hs_zero(
    thrust::device_vector<T>& y,
    thrust::device_vector<int>& rng_cones,
    thrust::device_vector<int>& idx_eq
) {
    int n_eq = idx_eq.size();
    int threads = std::min(n_eq, 1024);
    int blocks = (n_eq + threads - 1) / threads;

    _kernel_mul_Hs_zero<<<blocks, threads>>>(
        thrust::raw_pointer_cast(y.data()),
        thrust::raw_pointer_cast(rng_cones.data()),
        thrust::raw_pointer_cast(idx_eq.data()),
        n_eq
    );
    cudaDeviceSynchronize();
}

template <typename T>
__global__ void _kernel_affine_ds_zero(
    T* ds,
    int* rng_cones,
    int* idx_eq,
    int n_eq
) {
    int i = (blockIdx.x - 1) * blockDim.x + threadIdx.x;

    if (i <= n_eq) {
        ds[rng_cones[idx_eq[i]]] = static_cast<T>(0);
    }
}

template <typename T>
void affine_ds_zero(
    thrust::device_vector<T>& ds,
    thrust::device_vector<int>& rng_cones,
    thrust::device_vector<int>& idx_eq
) {
    int n_eq = idx_eq.size();
    int threads = std::min(n_eq, 1024);
    int blocks = (n_eq + threads - 1) / threads;

    _kernel_affine_ds_zero<<<blocks, threads>>>(
        thrust::raw_pointer_cast(ds.data()),
        thrust::raw_pointer_cast(rng_cones.data()),
        thrust::raw_pointer_cast(idx_eq.data()),
        n_eq
    );
    cudaDeviceSynchronize();
}

template <typename T>
__global__ void _kernel_combined_ds_shift_zero(
    T* shift,
    int* rng_cones,
    int* idx_eq,
    int n_eq
) {
    int i = (blockIdx.x - 1) * blockDim.x + threadIdx.x;

    if (i <= n_eq) {
        shift[rng_cones[idx_eq[i]]] = static_cast<T>(0);
    }
}

template <typename T>
void combined_ds_shift_zero(
    thrust::device_vector<T>& shift,
    thrust::device_vector<int>& rng_cones,
    thrust::device_vector<int>& idx_eq
) {
    int n_eq = idx_eq.size();
    int threads = std::min(n_eq, 1024);
    int blocks = (n_eq + threads - 1) / threads;

    _kernel_combined_ds_shift_zero<<<blocks, threads>>>(
        thrust::raw_pointer_cast(shift.data()),
        thrust::raw_pointer_cast(rng_cones.data()),
        thrust::raw_pointer_cast(idx_eq.data()),
        n_eq
    );
    cudaDeviceSynchronize();
}

template <typename T>
__global__ void _kernel_Δs_from_Δz_offset_zero(
    T* out,
    int* rng_cones,
    int* idx_eq,
    int n_eq
) {
    int i = (blockIdx.x - 1) * blockDim.x + threadIdx.x;

    if (i <= n_eq) {
        out[rng_cones[idx_eq[i]]] = static_cast<T>(0);
    }
}

template <typename T>
void Δs_from_Δz_offset_zero(
    thrust::device_vector<T>& out,
    thrust::device_vector<int>& rng_cones,
    thrust::device_vector<int>& idx_eq
) {
    int n_eq = idx_eq.size();
    int threads = std::min(n_eq, 1024);
    int blocks = (n_eq + threads - 1) / threads;

    _kernel_Δs_from_Δz_offset_zero<<<blocks, threads>>>(
        thrust::raw_pointer_cast(out.data()),
        thrust::raw_pointer_cast(rng_cones.data()),
        thrust::raw_pointer_cast(idx_eq.data()),
        n_eq
    );
    cudaDeviceSynchronize();
}
