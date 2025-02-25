#include <vector>
#include <algorithm>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
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

namespace Clarabel {

template <typename T>
class CompositeCone;

template <typename T>
T degree(const CompositeCone<T>& cones) {
    return cones.get_degree();
}

template <typename T>
T numel(const CompositeCone<T>& cones) {
    return cones.get_numel();
}

template <typename T>
bool is_symmetric(const CompositeCone<T>& cones) {
    return cones.is_symmetric();
}

template <typename T>
bool is_sparse_expandable(const CompositeCone<T>& cones) {
    throw std::runtime_error("This function should not be reachable");
}

template <typename T>
bool allows_primal_dual_scaling(const CompositeCone<T>& cones) {
    return std::all_of(cones.get_cones().begin(), cones.get_cones().end(), [](const auto& cone) {
        return allows_primal_dual_scaling(cone);
    });
}

template <typename T>
bool rectify_equilibration(
    CompositeCone<T>& cones,
    thrust::device_vector<T>& δ,
    thrust::device_vector<T>& e
) {
    bool any_changed = false;

    thrust::fill(δ.begin(), δ.end(), static_cast<T>(1));

    for (const auto& [cone, rng] : zip(cones.get_cones(), cones.get_rng_cones())) {
        auto δi = thrust::make_zip_iterator(thrust::make_tuple(δ.begin() + rng.first, δ.begin() + rng.second));
        auto ei = thrust::make_zip_iterator(thrust::make_tuple(e.begin() + rng.first, e.begin() + rng.second));
        any_changed |= rectify_equilibration(cone, δi, ei);
    }

    return any_changed;
}

template <typename T>
std::pair<T, T> margins(
    const CompositeCone<T>& cones,
    const thrust::device_vector<T>& z,
    PrimalOrDualCone pd
) {
    T α = std::numeric_limits<T>::max();
    T β = static_cast<T>(0);

    for (const auto& [cone, rng] : zip(cones.get_cones(), cones.get_rng_cones())) {
        auto z_view = thrust::make_zip_iterator(thrust::make_tuple(z.begin() + rng.first, z.begin() + rng.second));
        auto [αi, βi] = margins(cone, z_view, pd);
        α = std::min(α, αi);
        β += βi;
    }

    return {α, β};
}

template <typename T>
void scaled_unit_shift(
    CompositeCone<T>& cones,
    thrust::device_vector<T>& z,
    T α,
    PrimalOrDualCone pd
) {
    for (const auto& [cone, rng] : zip(cones.get_cones(), cones.get_rng_cones())) {
        auto z_view = thrust::make_zip_iterator(thrust::make_tuple(z.begin() + rng.first, z.begin() + rng.second));
        scaled_unit_shift(cone, z_view, α, pd);
    }
}

template <typename T>
void unit_initialization(
    CompositeCone<T>& cones,
    thrust::device_vector<T>& z,
    thrust::device_vector<T>& s
) {
    for (const auto& [cone, rng] : zip(cones.get_cones(), cones.get_rng_cones())) {
        auto z_view = thrust::make_zip_iterator(thrust::make_tuple(z.begin() + rng.first, z.begin() + rng.second));
        auto s_view = thrust::make_zip_iterator(thrust::make_tuple(s.begin() + rng.first, s.begin() + rng.second));
        unit_initialization(cone, z_view, s_view);
    }
}

template <typename T>
void set_identity_scaling(CompositeCone<T>& cones) {
    for (const auto& cone : cones.get_cones()) {
        set_identity_scaling(cone);
    }
}

template <typename T>
bool update_scaling(
    CompositeCone<T>& cones,
    thrust::device_vector<T>& s,
    thrust::device_vector<T>& z,
    T μ,
    ScalingStrategy scaling_strategy
) {
    for (const auto& [cone, rng] : zip(cones.get_cones(), cones.get_rng_cones())) {
        auto s_view = thrust::make_zip_iterator(thrust::make_tuple(s.begin() + rng.first, s.begin() + rng.second));
        auto z_view = thrust::make_zip_iterator(thrust::make_tuple(z.begin() + rng.first, z.begin() + rng.second));
        if (!update_scaling(cone, s_view, z_view, μ, scaling_strategy)) {
            return false;
        }
    }
    return true;
}

template <typename T>
void get_Hs(
    CompositeCone<T>& cones,
    thrust::device_vector<T>& Hsblock,
    bool is_triangular
) {
    for (const auto& [cone, rng] : zip(cones.get_cones(), cones.get_rng_blocks())) {
        auto Hsblock_view = thrust::make_zip_iterator(thrust::make_tuple(Hsblock.begin() + rng.first, Hsblock.begin() + rng.second));
        get_Hs(cone, Hsblock_view, is_triangular);
    }
}

template <typename T>
void mul_Hs(
    CompositeCone<T>& cones,
    thrust::device_vector<T>& y,
    const thrust::device_vector<T>& x,
    thrust::device_vector<T>& work
) {
    for (const auto& [cone, rng] : zip(cones.get_cones(), cones.get_rng_cones())) {
        auto y_view = thrust::make_zip_iterator(thrust::make_tuple(y.begin() + rng.first, y.begin() + rng.second));
        auto x_view = thrust::make_zip_iterator(thrust::make_tuple(x.begin() + rng.first, x.begin() + rng.second));
        auto work_view = thrust::make_zip_iterator(thrust::make_tuple(work.begin() + rng.first, work.begin() + rng.second));
        mul_Hs(cone, y_view, x_view, work_view);
    }
}

template <typename T>
void affine_ds(
    CompositeCone<T>& cones,
    thrust::device_vector<T>& ds,
    const thrust::device_vector<T>& s
) {
    for (const auto& [cone, rng] : zip(cones.get_cones(), cones.get_rng_cones())) {
        auto ds_view = thrust::make_zip_iterator(thrust::make_tuple(ds.begin() + rng.first, ds.begin() + rng.second));
        auto s_view = thrust::make_zip_iterator(thrust::make_tuple(s.begin() + rng.first, s.begin() + rng.second));
        affine_ds(cone, ds_view, s_view);
    }
}

template <typename T>
void combined_ds_shift(
    CompositeCone<T>& cones,
    thrust::device_vector<T>& shift,
    const thrust::device_vector<T>& step_z,
    const thrust::device_vector<T>& step_s,
    const thrust::device_vector<T>& z,
    T σμ
) {
    for (const auto& [cone, rng] : zip(cones.get_cones(), cones.get_rng_cones())) {
        auto shift_view = thrust::make_zip_iterator(thrust::make_tuple(shift.begin() + rng.first, shift.begin() + rng.second));
        auto step_z_view = thrust::make_zip_iterator(thrust::make_tuple(step_z.begin() + rng.first, step_z.begin() + rng.second));
        auto step_s_view = thrust::make_zip_iterator(thrust::make_tuple(step_s.begin() + rng.first, step_s.begin() + rng.second));
        combined_ds_shift(cone, shift_view, step_z_view, step_s_view, σμ);
    }
}

template <typename T>
void Δs_from_Δz_offset(
    CompositeCone<T>& cones,
    thrust::device_vector<T>& out,
    const thrust::device_vector<T>& ds,
    thrust::device_vector<T>& work,
    const thrust::device_vector<T>& z
) {
    for (const auto& [cone, rng] : zip(cones.get_cones(), cones.get_rng_cones())) {
        auto out_view = thrust::make_zip_iterator(thrust::make_tuple(out.begin() + rng.first, out.begin() + rng.second));
        auto ds_view = thrust::make_zip_iterator(thrust::make_tuple(ds.begin() + rng.first, ds.begin() + rng.second));
        auto work_view = thrust::make_zip_iterator(thrust::make_tuple(work.begin() + rng.first, work.begin() + rng.second));
        auto z_view = thrust::make_zip_iterator(thrust::make_tuple(z.begin() + rng.first, z.begin() + rng.second));
        Δs_from_Δz_offset(cone, out_view, ds_view, work_view, z_view);
    }
}

template <typename T>
std::pair<T, T> step_length(
    const CompositeCone<T>& cones,
    const thrust::device_vector<T>& dz,
    const thrust::device_vector<T>& ds,
    const thrust::device_vector<T>& z,
    const thrust::device_vector<T>& s,
    const Settings<T>& settings,
    T αmax
) {
    T α = αmax;

    auto innerfcn = [&](T α, bool symcond) {
        for (const auto& [cone, rng] : zip(cones.get_cones(), cones.get_rng_cones())) {
            if (is_symmetric(cone) == symcond) {
                continue;
            }
            auto dz_view = thrust::make_zip_iterator(thrust::make_tuple(dz.begin() + rng.first, dz.begin() + rng.second));
            auto ds_view = thrust::make_zip_iterator(thrust::make_tuple(ds.begin() + rng.first, ds.begin() + rng.second));
            auto z_view = thrust::make_zip_iterator(thrust::make_tuple(z.begin() + rng.first, z.begin() + rng.second));
            auto s_view = thrust::make_zip_iterator(thrust::make_tuple(s.begin() + rng.first, s.begin() + rng.second));
            auto [nextαz, nextαs] = step_length(cone, dz_view, ds_view, z_view, s_view, settings, α);
            α = std::min({α, nextαz, nextαs});
        }
        return α;
    };

    α = innerfcn(α, false);

    if (!is_symmetric(cones)) {
        α = std::min(α, settings.max_step_fraction);
    }

    α = innerfcn(α, true);

    return {α, α};
}

template <typename T>
T compute_barrier(
    const CompositeCone<T>& cones,
    const thrust::device_vector<T>& z,
    const thrust::device_vector<T>& s,
    const thrust::device_vector<T>& dz,
    const thrust::device_vector<T>& ds,
    T α
) {
    T barrier = static_cast<T>(0);

    for (const auto& [cone, rng] : zip(cones.get_cones(), cones.get_rng_cones())) {
        auto z_view = thrust::make_zip_iterator(thrust::make_tuple(z.begin() + rng.first, z.begin() + rng.second));
        auto s_view = thrust::make_zip_iterator(thrust::make_tuple(s.begin() + rng.first, s.begin() + rng.second));
        auto dz_view = thrust::make_zip_iterator(thrust::make_tuple(dz.begin() + rng.first, dz.begin() + rng.second));
        auto ds_view = thrust::make_zip_iterator(thrust::make_tuple(ds.begin() + rng.first, ds.begin() + rng.second));
        barrier += compute_barrier(cone, z_view, s_view, dz_view, ds_view, α);
    }

    return barrier;
}

template <typename T>
bool check_neighborhood(
    const CompositeCone<T>& cones,
    const thrust::device_vector<T>& z,
    const thrust::device_vector<T>& s,
    const thrust::device_vector<T>& dz,
    const thrust::device_vector<T>& ds,
    T α,
    T μ,
    T thr
) {
    bool centrality = true;

    for (const auto& [cone, rng] : zip(cones.get_cones(), cones.get_rng_cones())) {
        auto z_view = thrust::make_zip_iterator(thrust::make_tuple(z.begin() + rng.first, z.begin() + rng.second));
        auto s_view = thrust::make_zip_iterator(thrust::make_tuple(s.begin() + rng.first, s.begin() + rng.second));
        auto dz_view = thrust::make_zip_iterator(thrust::make_tuple(dz.begin() + rng.first, dz.begin() + rng.second));
        auto ds_view = thrust::make_zip_iterator(thrust::make_tuple(ds.begin() + rng.first, ds.begin() + rng.second));
        centrality = check_neighborhood(cone, z_view, s_view, dz_view, ds_view, α, μ, thr);
        if (!centrality) {
            return false;
        }
    }

    return true;
}

} // namespace Clarabel
