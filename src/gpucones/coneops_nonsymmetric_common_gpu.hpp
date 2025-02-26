#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cmath>
#include <cassert>

template <typename T>
__global__ void _kernel_mul_Hs_nonsymmetric(
    T* y,
    T* Hs,
    T* x,
    int* rng_cones,
    int n_shift,
    int n_nonsymmetric
) {
    int i = (blockIdx.x - 1) * blockDim.x + threadIdx.x;

    if (i <= n_nonsymmetric) {
        int shift_i = i + n_shift;
        int rng_i = rng_cones[shift_i];
        T* Hsi = &Hs[i * 9]; // 3x3 matrix flattened
        T* yi = &y[rng_i * 3];
        T* xi = &x[rng_i * 3];

        for (int j = 0; j < 3; ++j) {
            yi[j] = Hsi[j * 3] * xi[0] + Hsi[j * 3 + 1] * xi[1] + Hsi[j * 3 + 2] * xi[2];
        }
    }
}

template <typename T>
void mul_Hs_nonsymmetric(
    thrust::device_vector<T>& y,
    thrust::device_vector<T>& Hs,
    thrust::device_vector<T>& x,
    thrust::device_vector<int>& rng_cones,
    int n_shift,
    int n_nonsymmetric
) {
    int threads = min(n_nonsymmetric, 256);
    int blocks = (n_nonsymmetric + threads - 1) / threads;

    _kernel_mul_Hs_nonsymmetric<<<blocks, threads>>>(
        thrust::raw_pointer_cast(y.data()),
        thrust::raw_pointer_cast(Hs.data()),
        thrust::raw_pointer_cast(x.data()),
        thrust::raw_pointer_cast(rng_cones.data()),
        n_shift,
        n_nonsymmetric
    );

    cudaDeviceSynchronize();
}

template <typename T>
void affine_ds_nonsymmetric(
    thrust::device_vector<T>& ds,
    thrust::device_vector<T>& s,
    thrust::device_vector<int>& rng_cones,
    int n_shift,
    int n_nonsymmetric
) {
    int start = rng_cones[n_shift];
    int stop = rng_cones[n_shift + n_nonsymmetric];
    thrust::copy(s.begin() + start, s.begin() + stop + 1, ds.begin() + start);
}

template <typename T>
void Δs_from_Δz_offset_nonsymmetric(
    thrust::device_vector<T>& out,
    thrust::device_vector<T>& ds,
    thrust::device_vector<int>& rng_cones,
    int n_shift,
    int n_nonsymmetric
) {
    int start = rng_cones[n_shift];
    int stop = rng_cones[n_shift + n_nonsymmetric];
    thrust::copy(ds.begin() + start, ds.begin() + stop + 1, out.begin() + start);
}

template <typename T>
void use_dual_scaling_gpu(
    thrust::device_vector<T>& Hs,
    thrust::device_vector<T>& H_dual,
    T μ
) {
    thrust::transform(H_dual.begin(), H_dual.end(), Hs.begin(), [μ] __device__(T val) { return μ * val; });
}

template <typename T>
void use_primal_dual_scaling_exp(
    thrust::device_vector<T>& s,
    thrust::device_vector<T>& z,
    thrust::device_vector<T>& grad,
    thrust::device_vector<T>& Hs,
    thrust::device_vector<T>& H_dual
) {
    thrust::device_vector<T> st = grad;
    thrust::device_vector<T> δs(3, 0);
    thrust::device_vector<T> tmp(3, 0);

    thrust::device_vector<T> zt = gradient_primal_exp(s);
    T dot_sz = _dot_xy_gpu(z, s, 1, 3);
    T μ = dot_sz / 3;
    T μt = _dot_xy_gpu(zt, st, 1, 3);

    thrust::device_vector<T> δz = tmp;
    thrust::transform(s.begin(), s.end(), st.begin(), δs.begin(), [μ] __device__(T si, T sti) { return si + μ * sti; });
    thrust::transform(z.begin(), z.end(), zt.begin(), δz.begin(), [μ] __device__(T zi, T zti) { return zi + μ * zti; });

    T dot_δsz = _dot_xy_gpu(δs, δz, 1, 3);
    T de1 = μ * μt - 1;
    T de2 = _dot_xHy_gpu(zt, H_dual, zt) - 3 * μt * μt;

    if (abs(de1) > sqrt(std::numeric_limits<T>::epsilon()) &&
        abs(de2) > std::numeric_limits<T>::epsilon() &&
        dot_sz > 0 &&
        dot_δsz > 0) {

        thrust::transform(st.begin(), st.end(), tmp.begin(), [μt] __device__(T sti) { return μt * sti; });
        thrust::transform(zt.begin(), zt.end(), H_dual.begin(), tmp.begin(), tmp.begin(), [] __device__(T zti, T H_duali) { return zti - H_duali; });

        thrust::copy(H_dual.begin(), H_dual.end(), Hs.begin());
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                Hs[i * 3 + j] -= st[i] * st[j] / 3 + tmp[i] * tmp[j] / de2;
            }
        }

        T t = μ * _normHF(Hs);
        assert(t > 0);

        thrust::device_vector<T> axis_z(3);
        axis_z[0] = z[1] * zt[2] - z[2] * zt[1];
        axis_z[1] = z[2] * zt[0] - z[0] * zt[2];
        axis_z[2] = z[0] * zt[1] - z[1] * zt[0];
        _normalize(axis_z);

        for (int i = 0; i < 3; ++i) {
            for (int j = i; j < 3; ++j) {
                Hs[i * 3 + j] = s[i] * s[j] / dot_sz + δs[i] * δs[j] / dot_δsz + t * axis_z[i] * axis_z[j];
            }
        }

        Hs[1 * 3 + 0] = Hs[0 * 3 + 1];
        Hs[2 * 3 + 0] = Hs[0 * 3 + 2];
        Hs[2 * 3 + 1] = Hs[1 * 3 + 2];
    } else {
        use_dual_scaling_gpu(Hs, H_dual, μ);
    }
}

template <typename T>
void use_primal_dual_scaling_pow(
    thrust::device_vector<T>& s,
    thrust::device_vector<T>& z,
    thrust::device_vector<T>& grad,
    thrust::device_vector<T>& Hs,
    thrust::device_vector<T>& H_dual,
    T α
) {
    thrust::device_vector<T> st = grad;
    thrust::device_vector<T> δs(3, 0);
    thrust::device_vector<T> tmp(3, 0);

    thrust::device_vector<T> zt = gradient_primal_pow(s, α);
    T dot_sz = _dot_xy_gpu(z, s, 1, 3);
    T μ = dot_sz / 3;
    T μt = _dot_xy_gpu(zt, st, 1, 3);

    thrust::device_vector<T> δz = tmp;
    thrust::transform(s.begin(), s.end(), st.begin(), δs.begin(), [μ] __device__(T si, T sti) { return si + μ * sti; });
    thrust::transform(z.begin(), z.end(), zt.begin(), δz.begin(), [μ] __device__(T zi, T zti) { return zi + μ * zti; });

    T dot_δsz = _dot_xy_gpu(δs, δz, 1, 3);
    T de1 = μ * μt - 1;
    T de2 = _dot_xHy_gpu(zt, H_dual, zt) - 3 * μt * μt;

    if (abs(de1) > sqrt(std::numeric_limits<T>::epsilon()) &&
        abs(de2) > std::numeric_limits<T>::epsilon() &&
        dot_sz > 0 &&
        dot_δsz > 0) {

        thrust::transform(st.begin(), st.end(), tmp.begin(), [μt] __device__(T sti) { return μt * sti; });
        thrust::transform(zt.begin(), zt.end(), H_dual.begin(), tmp.begin(), tmp.begin(), [] __device__(T zti, T H_duali) { return zti - H_duali; });

        thrust::copy(H_dual.begin(), H_dual.end(), Hs.begin());
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                Hs[i * 3 + j] -= st[i] * st[j] / 3 + tmp[i] * tmp[j] / de2;
            }
        }

        T t = μ * _normHF(Hs);
        assert(t > 0);

        thrust::device_vector<T> axis_z(3);
        axis_z[0] = z[1] * zt[2] - z[2] * zt[1];
        axis_z[1] = z[2] * zt[0] - z[0] * zt[2];
        axis_z[2] = z[0] * zt[1] - z[1] * zt[0];
        _normalize(axis_z);

        for (int i = 0; i < 3; ++i) {
            for (int j = i; j < 3; ++j) {
                Hs[i * 3 + j] = s[i] * s[j] / dot_sz + δs[i] * δs[j] / dot_δsz + t * axis_z[i] * axis_z[j];
            }
        }

        Hs[1 * 3 + 0] = Hs[0 * 3 + 1];
        Hs[2 * 3 + 0] = Hs[0 * 3 + 2];
        Hs[2 * 3 + 1] = Hs[1 * 3 + 2];
    } else {
        use_dual_scaling_gpu(Hs, H_dual, μ);
    }
}

template <typename T>
T _dot_xHy_gpu(
    thrust::device_vector<T>& x,
    thrust::device_vector<T>& H,
    thrust::device_vector<T>& y
) {
    T val = 0;
    for (int j = 0; j < 3; ++j) {
        val += x[j] * (H[j * 3] * y[0] + H[j * 3 + 1] * y[1] + H[j * 3 + 2] * y[2]);
    }
    return val;
}

template <typename T>
T _normHF(
    thrust::device_vector<T>& H
) {
    T val = 0;
    for (int j = 0; j < 9; ++j) {
        val += H[j] * H[j];
    }
    return sqrt(val);
}

template <typename T>
void _normalize(
    thrust::device_vector<T>& x
) {
    T val = 0;
    for (int j = 0; j < 3; ++j) {
        val += x[j] * x[j];
    }
    T normx = sqrt(val);
    for (int j = 0; j < 3; ++j) {
        x[j] /= normx;
    }
}
