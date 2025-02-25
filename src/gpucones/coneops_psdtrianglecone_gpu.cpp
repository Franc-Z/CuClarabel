#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
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
#include <thrust/iterator/zip_function.h>

template <typename T>
__global__ void _kernel_scaled_unit_shift_psd(
    T* z,
    T α,
    int* rng_cones,
    int psd_dim,
    int n_shift,
    int n_psd
) {
    int i = (blockIdx.x - 1) * blockDim.x + threadIdx.x;

    if (i <= n_psd) {
        int shift_i = i + n_shift;
        int rng_cone_i = rng_cones[shift_i];
        T* zi = &z[rng_cone_i];
        for (int k = 1; k <= psd_dim; ++k) {
            zi[triangular_index(k)] += α;
        }
    }
}

template <typename T>
void scaled_unit_shift_psd(
    thrust::device_vector<T>& z,
    T α,
    thrust::device_vector<int>& rng_cones,
    int psd_dim,
    int n_shift,
    int n_psd
) {
    int threads = std::min(n_psd, 1024);
    int blocks = (n_psd + threads - 1) / threads;

    _kernel_scaled_unit_shift_psd<<<blocks, threads>>>(
        thrust::raw_pointer_cast(z.data()),
        α,
        thrust::raw_pointer_cast(rng_cones.data()),
        psd_dim,
        n_shift,
        n_psd
    );
    cudaDeviceSynchronize();
}

template <typename T>
void unit_initialization_psd(
    thrust::device_vector<T>& z,
    thrust::device_vector<T>& s,
    thrust::device_vector<int>& rng_cones,
    int psd_dim,
    int n_shift,
    int n_psd
) {
    int start = rng_cones[n_shift];
    int end = rng_cones[n_shift + n_psd];
    thrust::fill(z.begin() + start, z.begin() + end, T(0));
    thrust::fill(s.begin() + start, s.begin() + end, T(0));

    T α = T(1);
    scaled_unit_shift_psd(z, α, rng_cones, psd_dim, n_shift, n_psd);
    scaled_unit_shift_psd(s, α, rng_cones, psd_dim, n_shift, n_psd);
}

template <typename T>
__global__ void _kernel_set_identity_scaling_psd(
    T* R,
    T* Rinv,
    T* Hspsd,
    int psd_dim,
    int n_psd
) {
    int i = (blockIdx.x - 1) * blockDim.x + threadIdx.x;

    if (i <= n_psd) {
        for (int k = 0; k < psd_dim; ++k) {
            R[k * psd_dim + k + i * psd_dim * psd_dim] = T(1);
            Rinv[k * psd_dim + k + i * psd_dim * psd_dim] = T(1);
        }
        for (int k = 0; k < triangular_number(psd_dim); ++k) {
            Hspsd[k * triangular_number(psd_dim) + k + i * triangular_number(psd_dim) * triangular_number(psd_dim)] = T(1);
        }
    }
}

template <typename T>
void set_identity_scaling_psd(
    thrust::device_vector<T>& R,
    thrust::device_vector<T>& Rinv,
    thrust::device_vector<T>& Hspsd,
    int psd_dim,
    int n_psd
) {
    int threads = std::min(n_psd, 1024);
    int blocks = (n_psd + threads - 1) / threads;

    _kernel_set_identity_scaling_psd<<<blocks, threads>>>(
        thrust::raw_pointer_cast(R.data()),
        thrust::raw_pointer_cast(Rinv.data()),
        thrust::raw_pointer_cast(Hspsd.data()),
        psd_dim,
        n_psd
    );
    cudaDeviceSynchronize();
}

template <typename T>
void update_scaling_psd(
    thrust::device_vector<T>& L1,
    thrust::device_vector<T>& L2,
    thrust::device_vector<T>& z,
    thrust::device_vector<T>& s,
    thrust::device_vector<T>& workmat1,
    thrust::device_vector<T>& λpsd,
    thrust::device_vector<T>& Λisqrt,
    thrust::device_vector<T>& R,
    thrust::device_vector<T>& Rinv,
    thrust::device_vector<T>& Hspsd,
    thrust::device_vector<int>& rng_cones,
    int n_shift,
    int n_psd
) {
    svec_to_mat_gpu(L2, z, rng_cones, n_shift, n_psd);
    svec_to_mat_gpu(L1, s, rng_cones, n_shift, n_psd);

    int* infoz;
    int* infos;
    cusolverDnHandle_t handle;
    cusolverDnCreate(&handle);
    cusolverDnSpotrfBatched(handle, CUBLAS_FILL_MODE_LOWER, L2.size() / n_psd, thrust::raw_pointer_cast(L2.data()), L2.size() / n_psd, infoz, n_psd);
    cusolverDnSpotrfBatched(handle, CUBLAS_FILL_MODE_LOWER, L1.size() / n_psd, thrust::raw_pointer_cast(L1.data()), L1.size() / n_psd, infos, n_psd);
    cusolverDnDestroy(handle);

    mask_zeros(L2, 'U');
    mask_zeros(L1, 'U');

    if (!(thrust::all_of(infoz, infoz + n_psd, thrust::identity<int>()) && thrust::all_of(infos, infos + n_psd, thrust::identity<int>()))) {
        return false;
    }

    thrust::device_vector<T> tmp = workmat1;
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, L2.size() / n_psd, L1.size() / n_psd, L2.size() / n_psd, &one, thrust::raw_pointer_cast(L2.data()), L2.size() / n_psd, L2.size(), thrust::raw_pointer_cast(L1.data()), L1.size() / n_psd, L1.size(), &zero, thrust::raw_pointer_cast(tmp.data()), tmp.size() / n_psd, tmp.size(), n_psd);
    cublasDestroy(cublas_handle);

    thrust::device_vector<T> U, S, V;
    cusolverDnHandle_t svd_handle;
    cusolverDnCreate(&svd_handle);
    cusolverDnSgesvdj(svd_handle, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_MODE_VECTOR, tmp.size() / n_psd, tmp.size() / n_psd, thrust::raw_pointer_cast(tmp.data()), tmp.size() / n_psd, thrust::raw_pointer_cast(S.data()), thrust::raw_pointer_cast(U.data()), tmp.size() / n_psd, thrust::raw_pointer_cast(V.data()), tmp.size() / n_psd, nullptr, nullptr);
    cusolverDnDestroy(svd_handle);

    thrust::copy(S.begin(), S.end(), λpsd.begin());
    thrust::transform(λpsd.begin(), λpsd.end(), Λisqrt.begin(), thrust::placeholders::_1 = 1 / sqrt(thrust::placeholders::_1));

    cublasCreate(&cublas_handle);
    cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, L1.size() / n_psd, V.size() / n_psd, L1.size() / n_psd, &one, thrust::raw_pointer_cast(L1.data()), L1.size() / n_psd, L1.size(), thrust::raw_pointer_cast(V.data()), V.size() / n_psd, V.size(), &zero, thrust::raw_pointer_cast(R.data()), R.size() / n_psd, R.size(), n_psd);
    right_mul_batched(R, Λisqrt, R);

    cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, U.size() / n_psd, L2.size() / n_psd, U.size() / n_psd, &one, thrust::raw_pointer_cast(U.data()), U.size() / n_psd, U.size(), thrust::raw_pointer_cast(L2.data()), L2.size() / n_psd, L2.size(), &zero, thrust::raw_pointer_cast(Rinv.data()), Rinv.size() / n_psd, Rinv.size(), n_psd);
    left_mul_batched(Λisqrt, Rinv, Rinv);
    cublasDestroy(cublas_handle);

    thrust::device_vector<T> RRt = workmat1;
    thrust::fill(RRt.begin(), RRt.end(), T(0));
    cublasCreate(&cublas_handle);
    cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, R.size() / n_psd, R.size() / n_psd, R.size() / n_psd, &one, thrust::raw_pointer_cast(R.data()), R.size() / n_psd, R.size(), thrust::raw_pointer_cast(R.data()), R.size() / n_psd, R.size(), &zero, thrust::raw_pointer_cast(RRt.data()), RRt.size() / n_psd, RRt.size(), n_psd);
    cublasDestroy(cublas_handle);

    skron_batched(Hspsd, RRt);
}

template <typename T>
__global__ void _kernel_get_Hs_psd(
    T* Hsblock,
    T* Hs,
    int* rng_blocks,
    int n_shift,
    int n_psd
) {
    int i = (blockIdx.x - 1) * blockDim.x + threadIdx.x;

    if (i <= n_psd) {
        int shift_i = i + n_shift;
        int rng_i = rng_blocks[shift_i];
        T* Hsi = &Hs[i * 9];
        T* Hsblocki = &Hsblock[rng_i];

        for (int j = 0; j < 9; ++j) {
            Hsblocki[j] = Hsi[j];
        }
    }
}

template <typename T>
void get_Hs_psd(
    thrust::device_vector<T>& Hsblocks,
    thrust::device_vector<T>& Hs,
    thrust::device_vector<int>& rng_blocks,
    int n_shift,
    int n_psd
) {
    int threads = std::min(n_psd, 1024);
    int blocks = (n_psd + threads - 1) / threads;

    _kernel_get_Hs_psd<<<blocks, threads>>>(
        thrust::raw_pointer_cast(Hsblocks.data()),
        thrust::raw_pointer_cast(Hs.data()),
        thrust::raw_pointer_cast(rng_blocks.data()),
        n_shift,
        n_psd
    );
    cudaDeviceSynchronize();
}

template <typename T>
void combined_ds_shift_psd(
    CompositeConeGPU<T>& cones,
    thrust::device_vector<T>& shift,
    thrust::device_vector<T>& step_z,
    thrust::device_vector<T>& step_s,
    int n_shift,
    int n_psd,
    T σμ
) {
    thrust::device_vector<T> tmp = shift;
    thrust::device_vector<T>& R = cones.R;
    thrust::device_vector<T>& Rinv = cones.Rinv;
    thrust::device_vector<int>& rng_cones = cones.rng_cones;
    thrust::device_vector<T>& workmat1 = cones.workmat1;
    thrust::device_vector<T>& workmat2 = cones.workmat2;
    thrust::device_vector<T>& workmat3 = cones.workmat3;
    int psd_dim = cones.psd_dim;

    int start = rng_cones[n_shift];
    int end = rng_cones[n_shift + n_psd];
    thrust::copy(step_z.begin() + start, step_z.begin() + end, tmp.begin() + start);
    mul_Wx_psd(step_z, tmp, R, rng_cones, workmat1, workmat2, workmat3, n_shift, n_psd, false);

    thrust::copy(step_s.begin() + start, step_s.begin() + end, tmp.begin() + start);
    mul_WTx_psd(step_s, tmp, Rinv, rng_cones, workmat1, workmat2, workmat3, n_shift, n_psd, false);

    svec_to_mat_gpu(workmat1, step_z, rng_cones, n_shift, n_psd);
    svec_to_mat_gpu(workmat2, step_s, rng_cones, n_shift, n_psd);
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, workmat1.size() / n_psd, workmat2.size() / n_psd, workmat1.size() / n_psd, &one, thrust::raw_pointer_cast(workmat1.data()), workmat1.size() / n_psd, workmat1.size(), thrust::raw_pointer_cast(workmat2.data()), workmat2.size() / n_psd, workmat2.size(), &zero, thrust::raw_pointer_cast(workmat3.data()), workmat3.size() / n_psd, workmat3.size(), n_psd);
    cublasDestroy(cublas_handle);
    symmetric_part_gpu(workmat3);

    mat_to_svec_gpu(shift, workmat3, rng_cones, n_shift, n_psd);
    scaled_unit_shift_psd(shift, -σμ, rng_cones, psd_dim, n_shift, n_psd);
}

template <typename T>
__global__ void _kernel_op_λ(
    T* X,
    T* Z,
    T* λpsd,
    int psd_dim,
    int n_psd
) {
    int i = (blockIdx.x - 1) * blockDim.x + threadIdx.x;

    if (i <= n_psd) {
        T* Xi = &X[i * psd_dim * psd_dim];
        T* Zi = &Z[i * psd_dim * psd_dim];
        T* λi = &λpsd[i * psd_dim];
        for (int k = 0; k < psd_dim; ++k) {
            for (int j = 0; j < psd_dim; ++j) {
                Xi[k * psd_dim + j] = 2 * Zi[k * psd_dim + j] / (λi[k] + λi[j]);
            }
        }
    }
}

template <typename T>
void op_λ(
    thrust::device_vector<T>& X,
    thrust::device_vector<T>& Z,
    thrust::device_vector<T>& λpsd,
    int psd_dim,
    int n_psd
) {
    int threads = std::min(n_psd, 1024);
    int blocks = (n_psd + threads - 1) / threads;

    _kernel_op_λ<<<blocks, threads>>>(
        thrust::raw_pointer_cast(X.data()),
        thrust::raw_pointer_cast(Z.data()),
        thrust::raw_pointer_cast(λpsd.data()),
        psd_dim,
        n_psd
    );
    cudaDeviceSynchronize();
}

template <typename T>
void Δs_from_Δz_offset_psd(
    CompositeConeGPU<T>& cones,
    thrust::device_vector<T>& out,
    thrust::device_vector<T>& ds,
    thrust::device_vector<T>& work,
    int n_shift,
    int n_psd
) {
    thrust::device_vector<T>& R = cones.R;
    thrust::device_vector<T>& λpsd = cones.λpsd;
    thrust::device_vector<int>& rng_cones = cones.rng_cones;
    thrust::device_vector<T>& workmat1 = cones.workmat1;
    thrust::device_vector<T>& workmat2 = cones.workmat2;
    thrust::device_vector<T>& workmat3 = cones.workmat3;
    int psd_dim = cones.psd_dim;

    svec_to_mat_gpu(workmat2, ds, rng_cones, n_shift, n_psd);
    op_λ(workmat1, workmat2, λpsd, psd_dim, n_psd);
    mat_to_svec_gpu(work, workmat1, rng_cones, n_shift, n_psd);

    mul_WTx_psd(out, work, R, rng_cones, workmat1, workmat2, workmat3, n_shift, n_psd, false);
}

template <typename T>
T step_length_psd(
    thrust::device_vector<T>& dz,
    thrust::device_vector<T>& ds,
    thrust::device_vector<T>& Λisqrt,
    thrust::device_vector<T>& d,
    thrust::device_vector<T>& Rx,
    thrust::device_vector<T>& Rinv,
    thrust::device_vector<T>& workmat1,
    thrust::device_vector<T>& workmat2,
    thrust::device_vector<T>& workmat3,
    T αmax,
    thrust::device_vector<int>& rng_cones,
    int n_shift,
    int n_psd
) {
    thrust::device_vector<T>& workΔ = workmat1;

    mul_Wx_psd(d, dz, Rx, rng_cones, workmat1, workmat2, workmat3, n_shift, n_psd, true);
    T αz = step_length_psd_component_gpu(workΔ, d, Λisqrt, n_psd, αmax);

    mul_WTx_psd(d, ds, Rinv, rng_cones, workmat1, workmat2, workmat3, n_shift, n_psd, true);
    T αs = step_length_psd_component_gpu(workΔ, d, Λisqrt, n_psd, αmax);

    return std::min({αmax, αz, αs});
}

template <typename T>
__global__ void _kernel_logdet(
    T* barrier,
    T* fact,
    int psd_dim,
    int n_psd
) {
    int i = (blockIdx.x - 1) * blockDim.x + threadIdx.x;

    if (i <= n_psd) {
        T val = T(0);
        for (int k = 0; k < psd_dim; ++k) {
            val += logsafe(fact[k * psd_dim + k + i * psd_dim * psd_dim]);
        }
        barrier[i] = val + val;
    }
}

template <typename T>
T _logdet_barrier_psd(
    thrust::device_vector<T>& barrier,
    thrust::device_vector<T>& x,
    thrust::device_vector<T>& dx,
    T alpha,
    thrust::device_vector<T>& workmat1,
    thrust::device_vector<T>& workvec,
    thrust::device_vector<int>& rng,
    int psd_dim,
    int n_psd
) {
    thrust::device_vector<T>& Q = workmat1;
    thrust::device_vector<T>& q = workvec;

    thrust::transform(x.begin() + rng[0], x.begin() + rng[1], dx.begin() + rng[0], q.begin() + rng[0], thrust::placeholders::_1 + alpha * thrust::placeholders::_2);
    svec_to_mat_no_shift_gpu(Q, q, n_psd);

    int* info;
    cusolverDnHandle_t handle;
    cusolverDnCreate(&handle);
    cusolverDnSpotrfBatched(handle, CUBLAS_FILL_MODE_LOWER, Q.size() / n_psd, thrust::raw_pointer_cast(Q.data()), Q.size() / n_psd, info, n_psd);
    cusolverDnDestroy(handle);

    if (thrust::all_of(info, info + n_psd, thrust::identity<int>())) {
        int threads = std::min(n_psd, 1024);
        int blocks = (n_psd + threads - 1) / threads;

        _kernel_logdet<<<blocks, threads>>>(
            thrust::raw_pointer_cast(barrier.data()),
            thrust::raw_pointer_cast(Q.data()),
            psd_dim,
            n_psd
        );
        cudaDeviceSynchronize();

        return thrust::reduce(barrier.begin(), barrier.begin() + n_psd, T(0), thrust::plus<T>());
    } else {
        return std::numeric_limits<T>::max();
    }
}

template <typename T>
T compute_barrier_psd(
    thrust::device_vector<T>& barrier,
    thrust::device_vector<T>& z,
    thrust::device_vector<T>& s,
    thrust::device_vector<T>& dz,
    thrust::device_vector<T>& ds,
    T α,
    thrust::device_vector<T>& workmat1,
    thrust::device_vector<T>& workvec,
    thrust::device_vector<int>& rng_cones,
    int psd_dim,
    int n_shift,
    int n_psd
) {
    int start = rng_cones[n_shift];
    int end = rng_cones[n_shift + n_psd];

    T barrier_d = _logdet_barrier_psd(barrier, z, dz, α, workmat1, workvec, rng_cones, psd_dim, n_psd);
    T barrier_p = _logdet_barrier_psd(barrier, s, ds, α, workmat1, workvec, rng_cones, psd_dim, n_psd);

    return -barrier_d - barrier_p;
}

template <typename T>
void mul_Wx_psd(
    thrust::device_vector<T>& y,
    thrust::device_vector<T>& x,
    thrust::device_vector<T>& Rx,
    thrust::device_vector<int>& rng_cones,
    thrust::device_vector<T>& workmat1,
    thrust::device_vector<T>& workmat2,
    thrust::device_vector<T>& workmat3,
    int n_shift,
    int n_psd,
    bool step_search
) {
    thrust::device_vector<T>& X = workmat1;
    thrust::device_vector<T>& Y = workmat2;
    thrust::device_vector<T>& tmp = workmat3;

    svec_to_mat_gpu(X, x, rng_cones, n_shift, n_psd);

    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, Rx.size() / n_psd, X.size() / n_psd, Rx.size() / n_psd, &one, thrust::raw_pointer_cast(Rx.data()), Rx.size() / n_psd, Rx.size(), thrust::raw_pointer_cast(X.data()), X.size() / n_psd, X.size(), &zero, thrust::raw_pointer_cast(tmp.data()), tmp.size() / n_psd, tmp.size(), n_psd);
    cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, tmp.size() / n_psd, Rx.size() / n_psd, tmp.size() / n_psd, &one, thrust::raw_pointer_cast(tmp.data()), tmp.size() / n_psd, tmp.size(), thrust::raw_pointer_cast(Rx.data()), Rx.size() / n_psd, Rx.size(), &zero, thrust::raw_pointer_cast(Y.data()), Y.size() / n_psd, Y.size(), n_psd);
    cublasDestroy(cublas_handle);

    if (step_search) {
        mat_to_svec_no_shift_gpu(y, Y, n_psd);
    } else {
        mat_to_svec_gpu(y, Y, rng_cones, n_shift, n_psd);
    }
}

template <typename T>
void mul_WTx_psd(
    thrust::device_vector<T>& y,
    thrust::device_vector<T>& x,
    thrust::device_vector<T>& Rx,
    thrust::device_vector<int>& rng_cones,
    thrust::device_vector<T>& workmat1,
    thrust::device_vector<T>& workmat2,
    thrust::device_vector<T>& workmat3,
    int n_shift,
    int n_psd,
    bool step_search
) {
    thrust::device_vector<T>& X = workmat1;
    thrust::device_vector<T>& Y = workmat2;
    thrust::device_vector<T>& tmp = workmat3;

    svec_to_mat_gpu(X, x, rng_cones, n_shift, n_psd);

    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, X.size() / n_psd, Rx.size() / n_psd, X.size() / n_psd, &one, thrust::raw_pointer_cast(X.data()), X.size() / n_psd, X.size(), thrust::raw_pointer_cast(Rx.data()), Rx.size() / n_psd, Rx.size(), &zero, thrust::raw_pointer_cast(tmp.data()), tmp.size() / n_psd, tmp.size(), n_psd);
    cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, tmp.size() / n_psd, Rx.size() / n_psd, tmp.size() / n_psd, &one, thrust::raw_pointer_cast(Rx.data()), Rx.size() / n_psd, Rx.size(), thrust::raw_pointer_cast(tmp.data()), tmp.size() / n_psd, tmp.size(), &zero, thrust::raw_pointer_cast(Y.data()), Y.size() / n_psd, Y.size(), n_psd);
    cublasDestroy(cublas_handle);

    if (step_search) {
        mat_to_svec_no_shift_gpu(y, Y, n_psd);
    } else {
        mat_to_svec_gpu(y, Y, rng_cones, n_shift, n_psd);
    }
}

template <typename T>
T step_length_psd_component_gpu(
    thrust::device_vector<T>& workΔ,
    thrust::device_vector<T>& d,
    thrust::device_vector<T>& Λisqrt,
    int n_psd,
    T αmax
) {
    svec_to_mat_no_shift_gpu(workΔ, d, n_psd);
    lrscale_gpu(workΔ, Λisqrt);

    thrust::device_vector<T> e = cusolverDnSyevjBatched('N', 'U', workΔ);

    T γ = thrust::reduce(e.begin(), e.end(), std::numeric_limits<T>::max(), thrust::minimum<T>());
    if (γ < 0) {
        return std::min(1 / -γ, αmax);
    } else {
        return αmax;
    }
}

template <typename T>
__global__ void _kernel_svec_to_mat(
    T* Z,
    T* z,
    int* rng_blocks,
    int n_shift,
    int n_psd
) {
    int i = (blockIdx.x - 1) * blockDim.x + threadIdx.x;

    if (i <= n_psd) {
        int shift_i = i + n_shift;
        int rng_i = rng_blocks[shift_i];
        T* Zi = &Z[i * 9];
        T* zi = &z[rng_i];
        svec_to_mat(Zi, zi);
    }
}

template <typename T>
void svec_to_mat_gpu(
    thrust::device_vector<T>& Z,
    thrust::device_vector<T>& z,
    thrust::device_vector<int>& rng_blocks,
    int n_shift,
    int n_psd
) {
    int threads = std::min(n_psd, 1024);
    int blocks = (n_psd + threads - 1) / threads;

    _kernel_svec_to_mat<<<blocks, threads>>>(
        thrust::raw_pointer_cast(Z.data()),
        thrust::raw_pointer_cast(z.data()),
        thrust::raw_pointer_cast(rng_blocks.data()),
        n_shift,
        n_psd
    );
    cudaDeviceSynchronize();
}

template <typename T>
__global__ void _kernel_svec_to_mat_no_shift(
    T* Z,
    T* z,
    int n_psd
) {
    int i = (blockIdx.x - 1) * blockDim.x + threadIdx.x;

    if (i <= n_psd) {
        T* Zi = &Z[i * 9];
        int dim = sqrt(Zi.size());
        int rng_i = (i - 1) * triangular_number(dim);
        T* zi = &z[rng_i];
        svec_to_mat(Zi, zi);
    }
}

template <typename T>
void svec_to_mat_no_shift_gpu(
    thrust::device_vector<T>& Z,
    thrust::device_vector<T>& z,
    int n_psd
) {
    int threads = std::min(n_psd, 1024);
    int blocks = (n_psd + threads - 1) / threads;

    _kernel_svec_to_mat_no_shift<<<blocks, threads>>>(
        thrust::raw_pointer_cast(Z.data()),
        thrust::raw_pointer_cast(z.data()),
        n_psd
    );
    cudaDeviceSynchronize();
}

template <typename T>
__global__ void _kernel_mat_to_svec(
    T* z,
    T* Z,
    int* rng_blocks,
    int n_shift,
    int n_psd
) {
    int i = (blockIdx.x - 1) * blockDim.x + threadIdx.x;

    if (i <= n_psd) {
        int shift_i = i + n_shift;
        int rng_i = rng_blocks[shift_i];
        T* Zi = &Z[i * 9];
        T* zi = &z[rng_i];
        mat_to_svec(zi, Zi);
    }
}

template <typename T>
void mat_to_svec_gpu(
    thrust::device_vector<T>& z,
    thrust::device_vector<T>& Z,
    thrust::device_vector<int>& rng_blocks,
    int n_shift,
    int n_psd
) {
    int threads = std::min(n_psd, 1024);
    int blocks = (n_psd + threads - 1) / threads;

    _kernel_mat_to_svec<<<blocks, threads>>>(
        thrust::raw_pointer_cast(z.data()),
        thrust::raw_pointer_cast(Z.data()),
        thrust::raw_pointer_cast(rng_blocks.data()),
        n_shift,
        n_psd
    );
    cudaDeviceSynchronize();
}

template <typename T>
__global__ void _kernel_mat_to_svec_no_shift(
    T* z,
    T* Z,
    int n_psd
) {
    int i = (blockIdx.x - 1) * blockDim.x + threadIdx.x;

    if (i <= n_psd) {
        T* Zi = &Z[i * 9];
        int dim = sqrt(Zi.size());
        int rng_i = (i - 1) * triangular_number(dim);
        T* zi = &z[rng_i];
        mat_to_svec(zi, Zi);
    }
}

template <typename T>
void mat_to_svec_no_shift_gpu(
    thrust::device_vector<T>& z,
    thrust::device_vector<T>& Z,
    int n_psd
) {
    int threads = std::min(n_psd, 1024);
    int blocks = (n_psd + threads - 1) / threads;

    _kernel_mat_to_svec_no_shift<<<blocks, threads>>>(
        thrust::raw_pointer_cast(z.data()),
        thrust::raw_pointer_cast(Z.data()),
        n_psd
    );
    cudaDeviceSynchronize();
}

template <typename T>
void skron_full(
    thrust::device_vector<T>& out,
    thrust::device_vector<T>& A
) {
    T sqrt2 = sqrt(T(2));
    int n = sqrt(A.size());

    int col = 1;
    for (int l = 0; l < n; ++l) {
        for (int k = 0; k <= l; ++k) {
            int row = 1;
            bool kl_eq = (k == l);

            for (int j = 0; j < n; ++j) {
                T Ajl = A[j * n + l];
                T Ajk = A[j * n + k];

                for (int i = 0; i <= j; ++i) {
                    if (row > col) break;
                    bool ij_eq = (i == j);

                    if (!ij_eq && !kl_eq) {
                        out[row * n + col] = A[i * n + k] * Ajl + A[i * n + l] * Ajk;
                    } else if (ij_eq && !kl_eq) {
                        out[row * n + col] = sqrt2 * Ajl * Ajk;
                    } else if (!ij_eq && kl_eq) {
                        out[row * n + col] = sqrt2 * A[i * n + l] * Ajk;
                    } else {
                        out[row * n + col] = Ajl * Ajl;
                    }

                    out[col * n + row] = out[row * n + col];
                    ++row;
                }
            }
            ++col;
        }
    }
}

template <typename T>
__global__ void _kernel_skron(
    T* out,
    T* A,
    int n
) {
    int i = (blockIdx.x - 1) * blockDim.x + threadIdx.x;

    if (i <= n) {
        T* outi = &out[i * 9];
        T* Ai = &A[i * 9];
        skron_full(outi, Ai);
    }
}

template <typename T>
void skron_batched(
    thrust::device_vector<T>& out,
    thrust::device_vector<T>& A
) {
    int n = out.size() / 9;

    int threads = std::min(n, 1024);
    int blocks = (n + threads - 1) / threads;

    _kernel_skron<<<blocks, threads>>>(
        thrust::raw_pointer_cast(out.data()),
        thrust::raw_pointer_cast(A.data()),
        n
    );
    cudaDeviceSynchronize();
}

template <typename T>
__global__ void _kernel_right_mul(
    T* A,
    T* B,
    T* C,
    int n2,
    int n
) {
    int i = (blockIdx.x - 1) * blockDim.x + threadIdx.x;

    if (i <= n) {
        int k = i / n2;
        int j = i % n2;
        T val = B[j * n2 + k];
        for (int l = 0; l < n2; ++l) {
            C[l * n2 + j * n + k] = val * A[l * n2 + j * n + k];
        }
    }
}

template <typename T>
void right_mul_batched(
    thrust::device_vector<T>& A,
    thrust::device_vector<T>& B,
    thrust::device_vector<T>& C
) {
    int n2 = sqrt(A.size());
    int n = n2 * sqrt(A.size() / n2);

    int threads = std::min(n, 1024);
    int blocks = (n + threads - 1) / threads;

    _kernel_right_mul<<<blocks, threads>>>(
        thrust::raw_pointer_cast(A.data()),
        thrust::raw_pointer_cast(B.data()),
        thrust::raw_pointer_cast(C.data()),
        n2,
        n
    );
    cudaDeviceSynchronize();
}

template <typename T>
__global__ void _kernel_left_mul(
    T* A,
    T* B,
    T* C,
    int n2,
    int n
) {
    int i = (blockIdx.x - 1) * blockDim.x + threadIdx.x;

    if (i <= n) {
        int k = i / n2;
        int j = i % n2;
        T val = A[j * n2 + k];
        for (int l = 0; l < n2; ++l) {
            C[j * n2 + l * n + k] = val * B[j * n2 + l * n + k];
        }
    }
}

template <typename T>
void left_mul_batched(
    thrust::device_vector<T>& A,
    thrust::device_vector<T>& B,
    thrust::device_vector<T>& C
) {
    int n2 = sqrt(B.size());
    int n = n2 * sqrt(B.size() / n2);

    int threads = std::min(n, 1024);
    int blocks = (n + threads - 1) / threads;

    _kernel_left_mul<<<blocks, threads>>>(
        thrust::raw_pointer_cast(A.data()),
        thrust::raw_pointer_cast(B.data()),
        thrust::raw_pointer_cast(C.data()),
        n2,
        n
    );
    cudaDeviceSynchronize();
}
