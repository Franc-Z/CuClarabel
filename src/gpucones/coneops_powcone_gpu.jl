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
#include <thrust/iterator/zip_iterator.h>

template <typename T>
__global__ void _kernel_unit_initialization_pow(
    T* z,
    T* s,
    T* αp,
    int* rng_cones,
    int n_shift,
    int n_pow
) {
    int i = (blockIdx.x - 1) * blockDim.x + threadIdx.x;

    if (i <= n_pow) {
        int shift_i = i + n_shift;
        int rng_cone_i = rng_cones[shift_i];
        T* zi = &z[rng_cone_i];
        T* si = &s[rng_cone_i];
        si[0] = sqrt(static_cast<T>(1) + αp[i]);
        si[1] = sqrt(static_cast<T>(1) + (static_cast<T>(1) - αp[i]));
        si[2] = static_cast<T>(0);

        for (int j = 0; j < 3; ++j) {
            zi[j] = si[j];
        }
    }
}

template <typename T>
void unit_initialization_pow(
    thrust::device_vector<T>& z,
    thrust::device_vector<T>& s,
    thrust::device_vector<T>& αp,
    thrust::device_vector<int>& rng_cones,
    int n_shift,
    int n_pow
) {
    int threads = std::min(n_pow, 1024);
    int blocks = (n_pow + threads - 1) / threads;

    _kernel_unit_initialization_pow<<<blocks, threads>>>(
        thrust::raw_pointer_cast(z.data()),
        thrust::raw_pointer_cast(s.data()),
        thrust::raw_pointer_cast(αp.data()),
        thrust::raw_pointer_cast(rng_cones.data()),
        n_shift,
        n_pow
    );
    cudaDeviceSynchronize();
}

template <typename T>
void update_Hs_pow(
    thrust::device_vector<T>& s,
    thrust::device_vector<T>& z,
    thrust::device_vector<T>& grad,
    thrust::device_vector<T>& Hs,
    thrust::device_vector<T>& H_dual,
    T μ,
    int scaling_strategy,
    T α
) {
    if (scaling_strategy == 0) {
        use_dual_scaling_gpu(Hs, H_dual, μ);
    } else {
        use_primal_dual_scaling_pow(s, z, grad, Hs, H_dual, α);
    }
}

template <typename T>
__global__ void _kernel_update_scaling_pow(
    T* s,
    T* z,
    T* grad,
    T* Hs,
    T* H_dual,
    T* αp,
    int* rng_cones,
    T μ,
    int scaling_strategy,
    int n_shift,
    int n_exp,
    int n_pow
) {
    int i = (blockIdx.x - 1) * blockDim.x + threadIdx.x;

    if (i <= n_pow) {
        int shift_i = i + n_shift;
        int rng_i = rng_cones[shift_i];
        T* zi = &z[rng_i];
        T* si = &s[rng_i];
        int shift_exp = n_exp + i;
        T* gradi = &grad[shift_exp * 3];
        T* Hsi = &Hs[shift_exp * 9];
        T* Hi = &H_dual[shift_exp * 9];

        update_dual_grad_H_pow(gradi, Hi, zi, αp[i]);
        update_Hs_pow(si, zi, gradi, Hsi, Hi, μ, scaling_strategy, αp[i]);
    }
}

template <typename T>
void update_scaling_pow(
    thrust::device_vector<T>& s,
    thrust::device_vector<T>& z,
    thrust::device_vector<T>& grad,
    thrust::device_vector<T>& Hs,
    thrust::device_vector<T>& H_dual,
    thrust::device_vector<T>& αp,
    thrust::device_vector<int>& rng_cones,
    T μ,
    int scaling_strategy,
    int n_shift,
    int n_exp,
    int n_pow
) {
    int threads = std::min(n_pow, 1024);
    int blocks = (n_pow + threads - 1) / threads;

    _kernel_update_scaling_pow<<<blocks, threads>>>(
        thrust::raw_pointer_cast(s.data()),
        thrust::raw_pointer_cast(z.data()),
        thrust::raw_pointer_cast(grad.data()),
        thrust::raw_pointer_cast(Hs.data()),
        thrust::raw_pointer_cast(H_dual.data()),
        thrust::raw_pointer_cast(αp.data()),
        thrust::raw_pointer_cast(rng_cones.data()),
        μ,
        scaling_strategy,
        n_shift,
        n_exp,
        n_pow
    );
    cudaDeviceSynchronize();
}

template <typename T>
__global__ void _kernel_get_Hs_pow(
    T* Hsblock,
    T* Hs,
    int* rng_blocks,
    int n_shift,
    int n_exp,
    int n_pow
) {
    int i = (blockIdx.x - 1) * blockDim.x + threadIdx.x;

    if (i <= n_pow) {
        int shift_i = i + n_shift;
        int rng_i = rng_blocks[shift_i];
        int shift_exp = n_exp + i;
        T* Hsi = &Hs[shift_exp * 9];
        T* Hsblocki = &Hsblock[rng_i];

        for (int j = 0; j < 9; ++j) {
            Hsblocki[j] = Hsi[j];
        }
    }
}

template <typename T>
void get_Hs_pow(
    thrust::device_vector<T>& Hsblocks,
    thrust::device_vector<T>& Hs,
    thrust::device_vector<int>& rng_blocks,
    int n_shift,
    int n_exp,
    int n_pow
) {
    int threads = std::min(n_pow, 1024);
    int blocks = (n_pow + threads - 1) / threads;

    _kernel_get_Hs_pow<<<blocks, threads>>>(
        thrust::raw_pointer_cast(Hsblocks.data()),
        thrust::raw_pointer_cast(Hs.data()),
        thrust::raw_pointer_cast(rng_blocks.data()),
        n_shift,
        n_exp,
        n_pow
    );
    cudaDeviceSynchronize();
}

template <typename T>
__global__ void _kernel_combined_ds_shift_pow(
    T* shift,
    T* step_z,
    T* step_s,
    T* z,
    T* grad,
    T* H_dual,
    T* αp,
    int* rng_cones,
    T σμ,
    int n_shift,
    int n_exp,
    int n_pow
) {
    int i = (blockIdx.x - 1) * blockDim.x + threadIdx.x;

    if (i <= n_pow) {
        int shift_i = i + n_shift;
        int rng_i = rng_cones[shift_i];
        int shift_exp = n_exp + i;
        T* Hi = &H_dual[shift_exp * 9];
        T* gradi = &grad[shift_exp * 3];
        T* zi = &z[rng_i];
        T* step_si = &step_s[rng_i];
        T* step_zi = &step_z[rng_i];
        T* shifti = &shift[rng_i];

        T η[3] = {0, 0, 0};

        higher_correction_pow(Hi, zi, η, step_si, step_zi, αp[i]);

        for (int j = 0; j < 3; ++j) {
            shifti[j] = gradi[j] * σμ - η[j];
        }
    }
}

template <typename T>
void combined_ds_shift_pow(
    thrust::device_vector<T>& shift,
    thrust::device_vector<T>& step_z,
    thrust::device_vector<T>& step_s,
    thrust::device_vector<T>& z,
    thrust::device_vector<T>& grad,
    thrust::device_vector<T>& H_dual,
    thrust::device_vector<T>& αp,
    thrust::device_vector<int>& rng_cones,
    T σμ,
    int n_shift,
    int n_exp,
    int n_pow
) {
    int threads = std::min(n_pow, 1024);
    int blocks = (n_pow + threads - 1) / threads;

    _kernel_combined_ds_shift_pow<<<blocks, threads>>>(
        thrust::raw_pointer_cast(shift.data()),
        thrust::raw_pointer_cast(step_z.data()),
        thrust::raw_pointer_cast(step_s.data()),
        thrust::raw_pointer_cast(z.data()),
        thrust::raw_pointer_cast(grad.data()),
        thrust::raw_pointer_cast(H_dual.data()),
        thrust::raw_pointer_cast(αp.data()),
        thrust::raw_pointer_cast(rng_cones.data()),
        σμ,
        n_shift,
        n_exp,
        n_pow
    );
    cudaDeviceSynchronize();
}

template <typename T>
__global__ void _kernel_step_length_pow(
    T* dz,
    T* ds,
    T* z,
    T* s,
    T* α,
    T* αp,
    int* rng_cones,
    T αmax,
    T αmin,
    T step,
    int n_shift,
    int n_pow
) {
    int i = (blockIdx.x - 1) * blockDim.x + threadIdx.x;

    if (i <= n_pow) {
        int shift_i = i + n_shift;
        int rng_i = rng_cones[shift_i];
        T* dzi = &dz[rng_i];
        T* dsi = &ds[rng_i];
        T* zi = &z[rng_i];
        T* si = &s[rng_i];

        α[i] = backtrack_search_pow(dzi, zi, dsi, si, αmax, αmin, step, αp[i]);
    }
}

template <typename T>
T step_length_pow(
    thrust::device_vector<T>& dz,
    thrust::device_vector<T>& ds,
    thrust::device_vector<T>& z,
    thrust::device_vector<T>& s,
    thrust::device_vector<T>& α,
    thrust::device_vector<T>& αp,
    thrust::device_vector<int>& rng_cones,
    T αmax,
    T αmin,
    T step,
    int n_shift,
    int n_pow
) {
    int threads = std::min(n_pow, 1024);
    int blocks = (n_pow + threads - 1) / threads;

    _kernel_step_length_pow<<<blocks, threads>>>(
        thrust::raw_pointer_cast(dz.data()),
        thrust::raw_pointer_cast(ds.data()),
        thrust::raw_pointer_cast(z.data()),
        thrust::raw_pointer_cast(s.data()),
        thrust::raw_pointer_cast(α.data()),
        thrust::raw_pointer_cast(αp.data()),
        thrust::raw_pointer_cast(rng_cones.data()),
        αmax,
        αmin,
        step,
        n_shift,
        n_pow
    );
    cudaDeviceSynchronize();

    return thrust::reduce(α.begin(), α.begin() + n_pow, αmax, thrust::minimum<T>());
}

template <typename T>
T backtrack_search_pow(
    T* dz,
    T* z,
    T* ds,
    T* s,
    T α_init,
    T α_min,
    T step,
    T αp
) {
    T α = α_init;
    T work[3] = {0, 0, 0};

    while (true) {
        for (int i = 0; i < 3; ++i) {
            work[i] = z[i] + α * dz[i];
        }

        if (is_dual_feasible_pow(work, αp)) {
            break;
        }
        if ((α *= step) < α_min) {
            return 0;
        }
    }

    while (true) {
        for (int i = 0; i < 3; ++i) {
            work[i] = s[i] + α * ds[i];
        }

        if (is_primal_feasible_pow(work, αp)) {
            break;
        }
        if ((α *= step) < α_min) {
            return 0;
        }
    }

    return α;
}

template <typename T>
__global__ void _kernel_compute_barrier_pow(
    T* barrier,
    T* z,
    T* s,
    T* dz,
    T* ds,
    T α,
    T* αp,
    int* rng_cones,
    int n_shift,
    int n_pow
) {
    int i = (blockIdx.x - 1) * blockDim.x + threadIdx.x;

    if (i <= n_pow) {
        int shift_i = i + n_shift;
        int rng_i = rng_cones[shift_i];
        T* dzi = &dz[rng_i];
        T* dsi = &ds[rng_i];
        T* zi = &z[rng_i];
        T* si = &s[rng_i];

        T cur_z[3] = {zi[0] + α * dzi[0], zi[1] + α * dzi[1], zi[2] + α * dzi[2]};
        T cur_s[3] = {si[0] + α * dsi[0], si[1] + α * dsi[1], si[2] + α * dsi[2]};

        T barrier_d = barrier_dual_pow(cur_z, αp[i]);
        T barrier_p = barrier_primal_pow(cur_s, αp[i]);
        barrier[i] = barrier_d + barrier_p;
    }
}

template <typename T>
T compute_barrier_pow(
    thrust::device_vector<T>& barrier,
    thrust::device_vector<T>& z,
    thrust::device_vector<T>& s,
    thrust::device_vector<T>& dz,
    thrust::device_vector<T>& ds,
    T α,
    thrust::device_vector<T>& αp,
    thrust::device_vector<int>& rng_cones,
    int n_shift,
    int n_pow
) {
    int threads = std::min(n_pow, 1024);
    int blocks = (n_pow + threads - 1) / threads;

    _kernel_compute_barrier_pow<<<blocks, threads>>>(
        thrust::raw_pointer_cast(barrier.data()),
        thrust::raw_pointer_cast(z.data()),
        thrust::raw_pointer_cast(s.data()),
        thrust::raw_pointer_cast(dz.data()),
        thrust::raw_pointer_cast(ds.data()),
        α,
        thrust::raw_pointer_cast(αp.data()),
        thrust::raw_pointer_cast(rng_cones.data()),
        n_shift,
        n_pow
    );
    cudaDeviceSynchronize();

    return thrust::reduce(barrier.begin(), barrier.begin() + n_pow, static_cast<T>(0), thrust::plus<T>());
}

template <typename T>
T barrier_dual_pow(T* z, T α) {
    return -log(-z[2] / z[0]) - log(z[1] - z[0] - z[0] * log(-z[2] / z[0]));
}

template <typename T>
T barrier_primal_pow(T* s, T α) {
    T ω = _wright_omega_gpu(1 - s[0] / s[1] - log(s[1] / s[2]));
    ω = (ω - 1) * (ω - 1) / ω;
    return -log(ω) - 2 * log(s[1]) - log(s[2]) - 3;
}

template <typename T>
bool is_primal_feasible_pow(T* s, T α) {
    if (s[2] > 0 && s[1] > 0) {
        T res = s[1] * log(s[2] / s[1]) - s[0];
        if (res > 0) {
            return true;
        }
    }
    return false;
}

template <typename T>
bool is_dual_feasible_pow(T* z, T α) {
    if (z[2] > 0 && z[0] < 0) {
        T res = z[1] - z[0] - z[0] * log(-z[2] / z[0]);
        if (res > 0) {
            return true;
        }
    }
    return false;
}

template <typename T>
void gradient_primal_pow(T* s, T* grad, T α) {
    T ω = _wright_omega_gpu(1 - s[0] / s[1] - log(s[1] / s[2]));
    grad[0] = 1 / ((ω - 1) * s[1]);
    grad[1] = grad[0] + grad[0] * log(ω * s[1] / s[2]) - 1 / s[1];
    grad[2] = ω / ((1 - ω) * s[2]);
}

template <typename T>
void higher_correction_pow(T* H, T* z, T* η, T* ds, T* v, T α) {
    T cholH[9] = {0};
    bool issuccess = cholesky_3x3_explicit_factor(cholH, H);
    if (!issuccess) {
        std::fill(η, η + 3, 0);
        return;
    }

    T u[3] = {0};
    cholesky_3x3_explicit_solve(cholH, ds, u);

    η[1] = 1;
    η[2] = -z[0] / z[2];
    η[0] = log(η[2]);

    T ψ = z[0] * η[0] - z[0] + z[1];

    T dotψu = thrust::inner_product(η, η + 3, u, static_cast<T>(0));
    T dotψv = thrust::inner_product(η, η + 3, v, static_cast<T>(0));

    T coef = ((u[0] * (v[0] / z[0] - v[2] / z[2]) + u[2] * (z[0] * v[2] / z[2] - v[0]) / z[2]) * ψ - 2 * dotψu * dotψv) / (ψ * ψ * ψ);
    for (int i = 0; i < 3; ++i) {
        η[i] *= coef;
    }

    T inv_ψ2 = 1 / (ψ * ψ);

    η[0] += (1 / ψ - 2 / z[0]) * u[0] * v[0] / (z[0] * z[0]) - u[2] * v[2] / (z[2] * z[2]) / ψ + dotψu * inv_ψ2 * (v[0] / z[0] - v[2] / z[2]) + dotψv * inv_ψ2 * (u[0] / z[0] - u[2] / z[2]);
    η[2] += 2 * (z[0] / ψ - 1) * u[2] * v[2] / (z[2] * z[2] * z[2]) - (u[2] * v[0] + u[0] * v[2]) / (z[2] * z[2]) / ψ + dotψu * inv_ψ2 * (z[0] * v[2] / (z[2] * z[2]) - v[0] / z[2]) + dotψv * inv_ψ2 * (z[0] * u[2] / (z[2] * z[2]) - u[0] / z[2]);

    for (int i = 0; i < 3; ++i) {
        η[i] /= 2;
    }
}

template <typename T>
void update_dual_grad_H_pow(T* grad, T* H, T* z, T α) {
    T l = log(-z[2] / z[0]);
    T r = -z[0] * l - z[0] + z[1];

    T c2 = 1 / r;

    grad[0] = c2 * l - 1 / z[0];
    grad[1] = -c2;
    grad[2] = (c2 * z[0] - 1) / z[2];

    H[0] = ((r * r - z[0] * r + l * l * z[0] * z[0]) / (r * z[0] * z[0] * r));
    H[1] = (-l / (r * r));
    H[3] = H[1];
    H[4] = (1 / (r * r));
    H[2] = ((z[1] - z[0]) / (r * r * z[2]));
    H[6] = H[2];
    H[5] = (-z[0] / (r * r * z[2]));
    H[7] = H[5];
    H[8] = ((r * r - z[0] * r + z[0] * z[0]) / (r * r * z[2] * z[2]));
}

template <typename T>
T _wright_omega_gpu(T z) {
    if (z < 0) {
        return std::numeric_limits<T>::infinity();
    }

    T w;
    if (z < 1 + M_PI) {
        T zm1 = z - 1;
        T p = zm1;
        w = 1 + 0.5 * p;
        p *= zm1;
        w += (1 / 16.0) * p;
        p *= zm1;
        w -= (1 / 192.0) * p;
        p *= zm1;
        w -= (1 / 3072.0) * p;
        p *= zm1;
        w += (13 / 61440.0) * p;
    } else {
        T logz = log(z);
        T zinv = 1 / z;
        w = z - logz;

        T q = logz * zinv;
        w += q;

        q *= zinv;
        w += q * (logz / 2 - 1);

        q *= zinv;
        w += q * (logz * logz / 3.0 - 1.5 * logz + 1);
    }

    T r = z - w - log(w);

    for (int i = 0; i < 2; ++i) {
        T wp1 = w + 1;
        T t = wp1 * (wp1 + (2 * r) / 3.0);
        w *= 1 + (r / wp1) * (t - 0.5 * r) / (t - r);
        r = (2 * w * w - 8 * w - 1) / (72.0 * std::pow(wp1, 6)) * std::pow(r, 4);
    }

    return w;
}
