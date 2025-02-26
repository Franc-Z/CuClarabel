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
__global__ void _kernel_margins_soc(
    T* z,
    T* alpha,
    int* rng_cones,
    int n_shift,
    int n_soc
) {
    int i = (blockIdx.x - 1) * blockDim.x + threadIdx.x;

    if (i <= n_soc) {
        int shift_i = i + n_shift;
        int rng_cone_i = rng_cones[shift_i];
        int size_i = rng_cone_i.size();
        T* zi = &z[rng_cone_i];

        T val = static_cast<T>(0);
        for (int j = 1; j < size_i; ++j) {
            val += zi[j] * zi[j];
        }
        alpha[i] = zi[0] - sqrt(val);
    }
}

template <typename T>
std::pair<T, T> margins_soc(
    thrust::device_vector<T>& z,
    thrust::device_vector<T>& alpha,
    thrust::device_vector<int>& rng_cones,
    int n_shift,
    int n_soc,
    T alpha_min
) {
    int threads = std::min(n_soc, 1024);
    int blocks = (n_soc + threads - 1) / threads;

    _kernel_margins_soc<<<blocks, threads>>>(
        thrust::raw_pointer_cast(z.data()),
        thrust::raw_pointer_cast(alpha.data()),
        thrust::raw_pointer_cast(rng_cones.data()),
        n_shift,
        n_soc
    );
    cudaDeviceSynchronize();

    thrust::device_vector<T> alpha_soc(alpha.begin(), alpha.begin() + n_soc);
    alpha_min = std::min(alpha_min, *thrust::min_element(alpha_soc.begin(), alpha_soc.end()));
    thrust::transform(alpha_soc.begin(), alpha_soc.end(), alpha_soc.begin(), thrust::placeholders::_1 = thrust::max(static_cast<T>(0), thrust::placeholders::_1));
    return std::make_pair(alpha_min, thrust::reduce(alpha_soc.begin(), alpha_soc.end(), static_cast<T>(0), thrust::plus<T>()));
}

template <typename T>
__global__ void _kernel_scaled_unit_shift_soc(
    T* z,
    T alpha,
    int* rng_cones,
    int n_shift,
    int n_soc
) {
    int i = (blockIdx.x - 1) * blockDim.x + threadIdx.x;

    if (i <= n_soc) {
        int shift_i = i + n_shift;
        int rng_cone_i = rng_cones[shift_i];
        T* zi = &z[rng_cone_i];
        zi[0] += alpha;
    }
}

template <typename T>
void scaled_unit_shift_soc(
    thrust::device_vector<T>& z,
    thrust::device_vector<int>& rng_cones,
    T alpha,
    int n_shift,
    int n_soc
) {
    int threads = std::min(n_soc, 1024);
    int blocks = (n_soc + threads - 1) / threads;

    _kernel_scaled_unit_shift_soc<<<blocks, threads>>>(
        thrust::raw_pointer_cast(z.data()),
        alpha,
        thrust::raw_pointer_cast(rng_cones.data()),
        n_shift,
        n_soc
    );
    cudaDeviceSynchronize();
}

template <typename T>
__global__ void _kernel_unit_initialization_soc(
    T* z,
    T* s,
    int* rng_cones,
    int n_linear,
    int n_soc
) {
    int i = (blockIdx.x - 1) * blockDim.x + threadIdx.x;

    if (i <= n_soc) {
        int shift_i = i + n_linear;
        int rng_cone_i = rng_cones[shift_i];
        T* zi = &z[rng_cone_i];
        T* si = &s[rng_cone_i];
        zi[0] = static_cast<T>(1);
        for (int j = 1; j < rng_cone_i.size(); ++j) {
            zi[j] = static_cast<T>(0);
        }

        si[0] = static_cast<T>(1);
        for (int j = 1; j < rng_cone_i.size(); ++j) {
            si[j] = static_cast<T>(0);
        }
    }
}

template <typename T>
void unit_initialization_soc(
    thrust::device_vector<T>& z,
    thrust::device_vector<T>& s,
    thrust::device_vector<int>& rng_cones,
    int n_shift,
    int n_soc
) {
    int threads = std::min(n_soc, 1024);
    int blocks = (n_soc + threads - 1) / threads;

    _kernel_unit_initialization_soc<<<blocks, threads>>>(
        thrust::raw_pointer_cast(z.data()),
        thrust::raw_pointer_cast(s.data()),
        thrust::raw_pointer_cast(rng_cones.data()),
        n_shift,
        n_soc
    );
    cudaDeviceSynchronize();
}

template <typename T>
__global__ void _kernel_set_identity_scaling_soc(
    T* w,
    T* eta,
    int* rng_cones,
    int n_linear,
    int n_soc
) {
    int i = (blockIdx.x - 1) * blockDim.x + threadIdx.x;

    if (i <= n_soc) {
        int shift_i = i + n_linear;
        int rng_cone_i = rng_cones[shift_i];
        T* wi = &w[rng_cone_i];
        wi[0] = static_cast<T>(1);
        for (int j = 1; j < rng_cone_i.size(); ++j) {
            wi[j] = static_cast<T>(0);
        }
        eta[i] = static_cast<T>(1);
    }
}

template <typename T>
void set_identity_scaling_soc(
    thrust::device_vector<T>& w,
    thrust::device_vector<T>& eta,
    thrust::device_vector<int>& rng_cones,
    int n_shift,
    int n_soc
) {
    int threads = std::min(n_soc, 1024);
    int blocks = (n_soc + threads - 1) / threads;

    _kernel_set_identity_scaling_soc<<<blocks, threads>>>(
        thrust::raw_pointer_cast(w.data()),
        thrust::raw_pointer_cast(eta.data()),
        thrust::raw_pointer_cast(rng_cones.data()),
        n_shift,
        n_soc
    );
    cudaDeviceSynchronize();
}

template <typename T>
__global__ void _kernel_update_scaling_soc(
    T* s,
    T* z,
    T* w,
    T* lambda,
    T* eta,
    int* rng_cones,
    int n_shift,
    int n_soc
) {
    int i = (blockIdx.x - 1) * blockDim.x + threadIdx.x;

    if (i <= n_soc) {
        int shift_i = i + n_shift;
        int rng_i = rng_cones[shift_i];
        T* zi = &z[rng_i];
        T* si = &s[rng_i];
        T* wi = &w[rng_i];
        T* lambda_i = &lambda[rng_i];

        T zscale = _sqrt_soc_residual_gpu(zi);
        T sscale = _sqrt_soc_residual_gpu(si);

        eta[i] = sqrt(sscale / zscale);

        for (int k = 0; k < rng_i.size(); ++k) {
            w[k] = s[k] / sscale;
        }

        wi[0] += zi[0] / zscale;

        for (int j = 1; j < rng_i.size(); ++j) {
            wi[j] -= zi[j] / zscale;
        }

        T wscale = _sqrt_soc_residual_gpu(wi);
        for (int j = 0; j < rng_i.size(); ++j) {
            wi[j] /= wscale;
        }

        T w1sq = static_cast<T>(0);
        for (int j = 1; j < rng_i.size(); ++j) {
            w1sq += wi[j] * wi[j];
        }
        wi[0] = sqrt(1 + w1sq);

        T gamma_i = static_cast<T>(0.5) * wscale;
        lambda_i[0] = gamma_i;

        T coef = static_cast<T>(1) / (si[0] / sscale + zi[0] / zscale + 2 * gamma_i);
        T c1 = (gamma_i + zi[0] / zscale) / sscale;
        T c2 = (gamma_i + si[0] / sscale) / zscale;
        for (int j = 1; j < rng_i.size(); ++j) {
            lambda_i[j] = coef * (c1 * si[j] + c2 * zi[j]);
        }
        for (int j = 0; j < rng_i.size(); ++j) {
            lambda_i[j] *= sqrt(sscale * zscale);
        }
    }
}

template <typename T>
void update_scaling_soc(
    thrust::device_vector<T>& s,
    thrust::device_vector<T>& z,
    thrust::device_vector<T>& w,
    thrust::device_vector<T>& lambda,
    thrust::device_vector<T>& eta,
    thrust::device_vector<int>& rng_cones,
    int n_shift,
    int n_soc
) {
    int threads = std::min(n_soc, 1024);
    int blocks = (n_soc + threads - 1) / threads;

    _kernel_update_scaling_soc<<<blocks, threads>>>(
        thrust::raw_pointer_cast(s.data()),
        thrust::raw_pointer_cast(z.data()),
        thrust::raw_pointer_cast(w.data()),
        thrust::raw_pointer_cast(lambda.data()),
        thrust::raw_pointer_cast(eta.data()),
        thrust::raw_pointer_cast(rng_cones.data()),
        n_shift,
        n_soc
    );
    cudaDeviceSynchronize();
}

template <typename T>
__global__ void _kernel_get_Hs_soc(
    T* Hsblocks,
    T* w,
    T* eta,
    int* rng_cones,
    int* rng_blocks,
    int n_linear,
    int n_soc
) {
    int i = (blockIdx.x - 1) * blockDim.x + threadIdx.x;

    if (i <= n_soc) {
        int shift_i = i + n_linear;
        int rng_cone_i = rng_cones[shift_i];
        int rng_block_i = rng_blocks[shift_i];
        T* wi = &w[rng_cone_i];
        T* Hsblocki = &Hsblocks[rng_block_i];

        int hidx = 0;
        for (int col = 0; col < rng_cone_i.size(); ++col) {
            T wcol = wi[col];
            for (int row = 0; row < rng_cone_i.size(); ++row) {
                Hsblocki[hidx] = 2 * wi[row] * wcol;
                ++hidx;
            }
        }
        Hsblocki[0] -= static_cast<T>(1);
        for (int ind = 1; ind < rng_cone_i.size(); ++ind) {
            Hsblocki[(ind - 1) * rng_cone_i.size() + ind] += static_cast<T>(1);
        }
        for (int j = 0; j < rng_cone_i.size() * rng_cone_i.size(); ++j) {
            Hsblocki[j] *= eta[i] * eta[i];
        }
    }
}

template <typename T>
void get_Hs_soc(
    thrust::device_vector<T>& Hsblocks,
    thrust::device_vector<T>& w,
    thrust::device_vector<T>& eta,
    thrust::device_vector<int>& rng_cones,
    thrust::device_vector<int>& rng_blocks,
    int n_shift,
    int n_soc
) {
    int threads = std::min(n_soc, 1024);
    int blocks = (n_soc + threads - 1) / threads;

    _kernel_get_Hs_soc<<<blocks, threads>>>(
        thrust::raw_pointer_cast(Hsblocks.data()),
        thrust::raw_pointer_cast(w.data()),
        thrust::raw_pointer_cast(eta.data()),
        thrust::raw_pointer_cast(rng_cones.data()),
        thrust::raw_pointer_cast(rng_blocks.data()),
        n_shift,
        n_soc
    );
    cudaDeviceSynchronize();
}

template <typename T>
__global__ void _kernel_mul_Hs_soc(
    T* y,
    T* x,
    T* w,
    T* eta,
    int* rng_cones,
    int n_linear,
    int n_soc
) {
    int i = (blockIdx.x - 1) * blockDim.x + threadIdx.x;

    if (i <= n_soc) {
        int shift_i = i + n_linear;
        int rng_cone_i = rng_cones[shift_i];
        T* xi = &x[rng_cone_i];
        T* yi = &y[rng_cone_i];
        T* wi = &w[rng_cone_i];

        T c = 2 * _dot_xy_gpu(wi, xi, 0, rng_cone_i.size());

        yi[0] = -xi[0] + c * wi[0];
        for (int j = 1; j < rng_cone_i.size(); ++j) {
            yi[j] = xi[j] + c * wi[j];
        }

        for (int j = 0; j < rng_cone_i.size(); ++j) {
            yi[j] *= eta[i] * eta[i];
        }
    }
}

template <typename T>
void mul_Hs_soc(
    thrust::device_vector<T>& y,
    thrust::device_vector<T>& x,
    thrust::device_vector<T>& w,
    thrust::device_vector<T>& eta,
    thrust::device_vector<int>& rng_cones,
    int n_shift,
    int n_soc
) {
    int threads = std::min(n_soc, 1024);
    int blocks = (n_soc + threads - 1) / threads;

    _kernel_mul_Hs_soc<<<blocks, threads>>>(
        thrust::raw_pointer_cast(y.data()),
        thrust::raw_pointer_cast(x.data()),
        thrust::raw_pointer_cast(w.data()),
        thrust::raw_pointer_cast(eta.data()),
        thrust::raw_pointer_cast(rng_cones.data()),
        n_shift,
        n_soc
    );
    cudaDeviceSynchronize();
}

template <typename T>
__global__ void _kernel_affine_ds_soc(
    T* ds,
    T* lambda,
    int* rng_cones,
    int n_linear,
    int n_soc
) {
    int i = (blockIdx.x - 1) * blockDim.x + threadIdx.x;

    if (i <= n_soc) {
        int shift_i = i + n_linear;
        int rng_cone_i = rng_cones[shift_i];
        T* dsi = &ds[rng_cone_i];
        T* lambda_i = &lambda[rng_cone_i];

        dsi[0] = static_cast<T>(0);
        for (int j = 0; j < rng_cone_i.size(); ++j) {
            dsi[0] += lambda_i[j] * lambda_i[j];
        }
        T lambda_i0 = lambda_i[0];
        for (int j = 1; j < rng_cone_i.size(); ++j) {
            dsi[j] = 2 * lambda_i0 * lambda_i[j];
        }
    }
}

template <typename T>
void affine_ds_soc(
    thrust::device_vector<T>& ds,
    thrust::device_vector<T>& lambda,
    thrust::device_vector<int>& rng_cones,
    int n_shift,
    int n_soc
) {
    int threads = std::min(n_soc, 1024);
    int blocks = (n_soc + threads - 1) / threads;

    _kernel_affine_ds_soc<<<blocks, threads>>>(
        thrust::raw_pointer_cast(ds.data()),
        thrust::raw_pointer_cast(lambda.data()),
        thrust::raw_pointer_cast(rng_cones.data()),
        n_shift,
        n_soc
    );
    cudaDeviceSynchronize();
}

template <typename T>
__global__ void _kernel_combined_ds_shift_soc(
    T* shift,
    T* step_z,
    T* step_s,
    T* w,
    T* eta,
    int* rng_cones,
    int n_linear,
    int n_soc,
    T sigma_mu
) {
    int i = (blockIdx.x - 1) * blockDim.x + threadIdx.x;

    if (i <= n_soc) {
        int shift_i = i + n_linear;
        int rng_cone_i = rng_cones[shift_i];
        T* step_zi = &step_z[rng_cone_i];
        T* step_si = &step_s[rng_cone_i];
        T* wi = &w[rng_cone_i];
        T* shifti = &shift[rng_cone_i];

        T* tmp = shifti;

        for (int j = 0; j < rng_cone_i.size(); ++j) {
            tmp[j] = step_zi[j];
        }
        T zeta = static_cast<T>(0);

        for (int j = 1; j < rng_cone_i.size(); ++j) {
            zeta += wi[j] * tmp[j];
        }

        T c = tmp[0] + zeta / (1 + wi[0]);

        step_zi[0] = eta[i] * (wi[0] * tmp[0] + zeta);

        for (int j = 1; j < rng_cone_i.size(); ++j) {
            step_zi[j] = eta[i] * (tmp[j] + c * wi[j]);
        }

        for (int j = 0; j < rng_cone_i.size(); ++j) {
            tmp[j] = step_si[j];
        }
        zeta = static_cast<T>(0);
        for (int j = 1; j < rng_cone_i.size(); ++j) {
            zeta += wi[j] * tmp[j];
        }

        c = -tmp[0] + zeta / (1 + wi[0]);

        step_si[0] = (static_cast<T>(1) / eta[i]) * (wi[0] * tmp[0] - zeta);

        for (int j = 1; j < rng_cone_i.size(); ++j) {
            step_si[j] = (static_cast<T>(1) / eta[i]) * (tmp[j] + c * wi[j]);
        }

        T val = static_cast<T>(0);
        for (int j = 0; j < rng_cone_i.size(); ++j) {
            val += step_si[j] * step_zi[j];
        }
        shifti[0] = val - sigma_mu;

        T s0 = step_si[0];
        T z0 = step_zi[0];
        for (int j = 1; j < rng_cone_i.size(); ++j) {
            shifti[j] = s0 * step_zi[j] + z0 * step_si[j];
        }
    }
}

template <typename T>
void combined_ds_shift_soc(
    thrust::device_vector<T>& shift,
    thrust::device_vector<T>& step_z,
    thrust::device_vector<T>& step_s,
    thrust::device_vector<T>& w,
    thrust::device_vector<T>& eta,
    thrust::device_vector<int>& rng_cones,
    int n_shift,
    int n_soc,
    T sigma_mu
) {
    int threads = std::min(n_soc, 1024);
    int blocks = (n_soc + threads - 1) / threads;

    _kernel_combined_ds_shift_soc<<<blocks, threads>>>(
        thrust::raw_pointer_cast(shift.data()),
        thrust::raw_pointer_cast(step_z.data()),
        thrust::raw_pointer_cast(step_s.data()),
        thrust::raw_pointer_cast(w.data()),
        thrust::raw_pointer_cast(eta.data()),
        thrust::raw_pointer_cast(rng_cones.data()),
        n_shift,
        n_soc,
        sigma_mu
    );
    cudaDeviceSynchronize();
}

template <typename T>
__global__ void _kernel_delta_s_from_delta_z_offset_soc(
    T* out,
    T* ds,
    T* z,
    T* w,
    T* lambda,
    T* eta,
    int* rng_cones,
    int n_shift,
    int n_soc
) {
    int i = (blockIdx.x - 1) * blockDim.x + threadIdx.x;

    if (i <= n_soc) {
        int shift_i = i + n_shift;
        int rng_cone_i = rng_cones[shift_i];
        T* outi = &out[rng_cone_i];
        T* dsi = &ds[rng_cone_i];
        T* zi = &z[rng_cone_i];
        T* wi = &w[rng_cone_i];
        T* lambda_i = &lambda[rng_cone_i];

        T reszi = _soc_residual_gpu(zi);

        T lambda1ds1 = _dot_xy_gpu(lambda_i, dsi, 1, rng_cone_i.size());
        T w1ds1 = _dot_xy_gpu(wi, dsi, 1, rng_cone_i.size());

        for (int j = 0; j < rng_cone_i.size(); ++j) {
            outi[j] = -zi[j];
        }
        outi[0] = zi[0];

        T c = lambda_i[0] * dsi[0] - lambda1ds1;
        for (int j = 0; j < rng_cone_i.size(); ++j) {
            outi[j] *= c / reszi;
        }

        outi[0] += eta[i] * w1ds1;
        for (int j = 1; j < rng_cone_i.size(); ++j) {
            outi[j] += eta[i] * (dsi[j] + w1ds1 / (1 + wi[0]) * wi[j]);
        }

        for (int j = 0; j < rng_cone_i.size(); ++j) {
            outi[j] *= static_cast<T>(1) / lambda_i[0];
        }
    }
}

template <typename T>
void delta_s_from_delta_z_offset_soc(
    thrust::device_vector<T>& out,
    thrust::device_vector<T>& ds,
    thrust::device_vector<T>& z,
    thrust::device_vector<T>& w,
    thrust::device_vector<T>& lambda,
    thrust::device_vector<T>& eta,
    thrust::device_vector<int>& rng_cones,
    int n_shift,
    int n_soc
) {
    int threads = std::min(n_soc, 1024);
    int blocks = (n_soc + threads - 1) / threads;

    _kernel_delta_s_from_delta_z_offset_soc<<<blocks, threads>>>(
        thrust::raw_pointer_cast(out.data()),
        thrust::raw_pointer_cast(ds.data()),
        thrust::raw_pointer_cast(z.data()),
        thrust::raw_pointer_cast(w.data()),
        thrust::raw_pointer_cast(lambda.data()),
        thrust::raw_pointer_cast(eta.data()),
        thrust::raw_pointer_cast(rng_cones.data()),
        n_shift,
        n_soc
    );
    cudaDeviceSynchronize();
}

template <typename T>
__global__ void _kernel_step_length_soc(
    T* dz,
    T* ds,
    T* z,
    T* s,
    T* alpha,
    int* rng_cones,
    int n_linear,
    int n_soc
) {
    int i = (blockIdx.x - 1) * blockDim.x + threadIdx.x;

    if (i <= n_soc) {
        int shift_i = i + n_linear;
        int rng_cone_i = rng_cones[shift_i];
        T* si = &s[rng_cone_i];
        T* dsi = &ds[rng_cone_i];
        T* zi = &z[rng_cone_i];
        T* dzi = &dz[rng_cone_i];

        T alpha_z = _step_length_soc_component_gpu(zi, dzi, alpha[i]);
        T alpha_s = _step_length_soc_component_gpu(si, dsi, alpha[i]);
        alpha[i] = std::min(alpha_z, alpha_s);
    }
}

template <typename T>
T step_length_soc(
    thrust::device_vector<T>& dz,
    thrust::device_vector<T>& ds,
    thrust::device_vector<T>& z,
    thrust::device_vector<T>& s,
    thrust::device_vector<T>& alpha,
    T alpha_max,
    thrust::device_vector<int>& rng_cones,
    int n_shift,
    int n_soc
) {
    int threads = std::min(n_soc, 1024);
    int blocks = (n_soc + threads - 1) / threads;

    _kernel_step_length_soc<<<blocks, threads>>>(
        thrust::raw_pointer_cast(dz.data()),
        thrust::raw_pointer_cast(ds.data()),
        thrust::raw_pointer_cast(z.data()),
        thrust::raw_pointer_cast(s.data()),
        thrust::raw_pointer_cast(alpha.data()),
        thrust::raw_pointer_cast(rng_cones.data()),
        n_shift,
        n_soc
    );
    cudaDeviceSynchronize();

    return std::min(alpha_max, *thrust::min_element(alpha.begin(), alpha.begin() + n_soc));
}

template <typename T>
__global__ void _kernel_compute_barrier_soc(
    T* barrier,
    T* z,
    T* s,
    T* dz,
    T* ds,
    T alpha,
    int* rng_cones,
    int n_linear,
    int n_soc
) {
    int i = (blockIdx.x - 1) * blockDim.x + threadIdx.x;

    if (i <= n_soc) {
        int shift_i = i + n_linear;
        int rng_cone_i = rng_cones[shift_i];
        T* si = &s[rng_cone_i];
        T* dsi = &ds[rng_cone_i];
        T* zi = &z[rng_cone_i];
        T* dzi = &dz[rng_cone_i];
        T res_si = _soc_residual_shifted(si, dsi, alpha);
        T res_zi = _soc_residual_shifted(zi, dzi, alpha);

        barrier[i] = (res_si > 0 && res_zi > 0) ? -log(res_si * res_zi) / 2 : std::numeric_limits<T>::infinity();
    }
}

template <typename T>
T compute_barrier_soc(
    thrust::device_vector<T>& barrier,
    thrust::device_vector<T>& z,
    thrust::device_vector<T>& s,
    thrust::device_vector<T>& dz,
    thrust::device_vector<T>& ds,
    T alpha,
    thrust::device_vector<int>& rng_cones,
    int n_linear,
    int n_soc
) {
    int threads = std::min(n_soc, 1024);
    int blocks = (n_soc + threads - 1) / threads;

    _kernel_compute_barrier_soc<<<blocks, threads>>>(
        thrust::raw_pointer_cast(barrier.data()),
        thrust::raw_pointer_cast(z.data()),
        thrust::raw_pointer_cast(s.data()),
        thrust::raw_pointer_cast(dz.data()),
        thrust::raw_pointer_cast(ds.data()),
        alpha,
        thrust::raw_pointer_cast(rng_cones.data()),
        n_linear,
        n_soc
    );
    cudaDeviceSynchronize();

    return thrust::reduce(barrier.begin(), barrier.begin() + n_soc, static_cast<T>(0), thrust::plus<T>());
}

template <typename T>
T _soc_residual_gpu(T* z, int size) {
    T res = z[0] * z[0];
    for (int j = 1; j < size; ++j) {
        res -= z[j] * z[j];
    }
    return res;
}

template <typename T>
T _sqrt_soc_residual_gpu(T* z, int size) {
    T res = _soc_residual_gpu(z, size);
    return res > 0 ? sqrt(res) : static_cast<T>(0);
}

template <typename T>
T _dot_xy_gpu(T* x, T* y, int start, int end) {
    T val = static_cast<T>(0);
    for (int j = start; j < end; ++j) {
        val += x[j] * y[j];
    }
    return val;
}

template <typename T>
void _minus_vec_gpu(T* y, T* x, int size) {
    for (int j = 0; j < size; ++j) {
        y[j] = -x[j];
    }
}

template <typename T>
void _multiply_gpu(T* x, T a, int size) {
    for (int j = 0; j < size; ++j) {
        x[j] *= a;
    }
}

template <typename T>
T _step_length_soc_component_gpu(
    T* x,
    T* y,
    T alpha_max,
    int size
) {
    T a = _soc_residual_gpu(y, size);
    T b = 2 * (x[0] * y[0] - _dot_xy_gpu(x, y, 1, size));
    T c = std::max(static_cast<T>(0), _soc_residual_gpu(x, size));
    T d = b * b - 4 * a * c;

    if (c < 0) {
        return -std::numeric_limits<T>::infinity();
    }

    if ((a > 0 && b > 0) || d < 0) {
        return alpha_max;
    }

    if (a == 0) {
        return alpha_max;
    }

    if (c == 0) {
        return (a >= 0 ? alpha_max : static_cast<T>(0));
    }

    T t = (b >= 0) ? (-b - sqrt(d)) : (-b + sqrt(d));

    T r1 = (2 * c) / t;
    T r2 = t / (2 * a);

    r1 = r1 < 0 ? std::numeric_limits<T>::max() : r1;
    r2 = r2 < 0 ? std::numeric_limits<T>::max() : r2;

    return std::min(alpha_max, std::min(r1, r2));
}
