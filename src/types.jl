#include <vector>
#include <optional>
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

namespace Clarabel {

template <typename T>
class PresolverRowReductionIndex {
public:
    std::vector<bool> keep_logical;

    PresolverRowReductionIndex(int size) : keep_logical(size, true) {}
};

template <typename T>
class Presolver {
public:
    std::vector<SupportedCone> init_cones;
    std::optional<PresolverRowReductionIndex> reduce_map;
    int mfull;
    int mreduced;
    T infbound;

    Presolver(const std::vector<SupportedCone>& init_cones, int mfull, int mreduced, T infbound)
        : init_cones(init_cones), mfull(mfull), mreduced(mreduced), infbound(infbound) {}
};

template <typename T>
class DefaultVariables {
public:
    thrust::device_vector<T> x, s, z;
    T τ, κ;

    DefaultVariables(int n, int m, bool use_gpu)
        : x(n), s(m), z(m), τ(1), κ(1) {
        if (!use_gpu) {
            x = thrust::host_vector<T>(n);
            s = thrust::host_vector<T>(m);
            z = thrust::host_vector<T>(m);
        }
    }
};

enum class ScalingStrategy {
    PrimalDual = 0,
    Dual = 1
};

template <typename T>
class DefaultResiduals {
public:
    thrust::device_vector<T> rx, rz, rx_inf, rz_inf, Px;
    T rτ, dot_qx, dot_bz, dot_sz, dot_xPx;

    DefaultResiduals(int n, int m, bool use_gpu)
        : rx(n), rz(m), rx_inf(n), rz_inf(m), Px(n), rτ(1), dot_qx(0), dot_bz(0), dot_sz(0), dot_xPx(0) {
        if (!use_gpu) {
            rx = thrust::host_vector<T>(n);
            rz = thrust::host_vector<T>(m);
            rx_inf = thrust::host_vector<T>(n);
            rz_inf = thrust::host_vector<T>(m);
            Px = thrust::host_vector<T>(n);
        }
    }
};

template <typename T>
class DefaultEquilibration {
public:
    std::vector<T> d, dinv, e, einv;
    T c;
    thrust::device_vector<T> d_gpu, dinv_gpu, e_gpu, einv_gpu;

    DefaultEquilibration(int n, int m, bool use_gpu)
        : d(n, 1), dinv(n, 1), e(m, 1), einv(m, 1), c(1) {
        if (use_gpu) {
            d_gpu = thrust::device_vector<T>(d.begin(), d.end());
            dinv_gpu = thrust::device_vector<T>(dinv.begin(), dinv.end());
            e_gpu = thrust::device_vector<T>(e.begin(), e.end());
            einv_gpu = thrust::device_vector<T>(einv.begin(), einv.end());
        }
    }
};

template <typename T>
class DefaultProblemData {
public:
    thrust::device_vector<T> P, q, A, b;
    std::vector<SupportedCone> cones;
    int n, m;
    DefaultEquilibration<T> equilibration;
    std::optional<T> normq, normb;
    std::optional<Presolver<T>> presolver;
    std::optional<ChordalInfo<T>> chordal_info;
    thrust::device_vector<T> P_gpu, q_gpu, A_gpu, At_gpu, b_gpu;

    DefaultProblemData(const thrust::device_vector<T>& P, const thrust::device_vector<T>& q, const thrust::device_vector<T>& A, const thrust::device_vector<T>& b, const std::vector<SupportedCone>& cones, int n, int m, const DefaultEquilibration<T>& equilibration)
        : P(P), q(q), A(A), b(b), cones(cones), n(n), m(m), equilibration(equilibration) {}
};

template <typename T>
class DefaultInfo {
public:
    T μ, sigma, step_length, cost_primal, cost_dual, res_primal, res_dual, res_primal_inf, res_dual_inf, gap_abs, gap_rel, ktratio;
    T prev_cost_primal, prev_cost_dual, prev_res_primal, prev_res_dual, prev_gap_abs, prev_gap_rel;
    double solve_time;
    SolverStatus status;
    unsigned int iterations;

    DefaultInfo()
        : μ(0), sigma(0), step_length(0), cost_primal(0), cost_dual(0), res_primal(0), res_dual(0), res_primal_inf(0), res_dual_inf(0), gap_abs(0), gap_rel(0), ktratio(0),
          prev_cost_primal(std::numeric_limits<T>::max()), prev_cost_dual(std::numeric_limits<T>::max()), prev_res_primal(std::numeric_limits<T>::max()), prev_res_dual(std::numeric_limits<T>::max()), prev_gap_abs(std::numeric_limits<T>::max()), prev_gap_rel(std::numeric_limits<T>::max()),
          solve_time(0), status(SolverStatus::UNSOLVED), iterations(0) {}
};

template <typename T>
class DefaultSolution {
public:
    thrust::device_vector<T> x, z, s;
    SolverStatus status;
    T obj_val, obj_val_dual, solve_time, r_prim, r_dual;
    unsigned int iterations;

    DefaultSolution(int n, int m, bool use_gpu)
        : x(n), z(m), s(m), status(SolverStatus::UNSOLVED), obj_val(std::numeric_limits<T>::quiet_NaN()), obj_val_dual(std::numeric_limits<T>::quiet_NaN()), solve_time(0), r_prim(std::numeric_limits<T>::quiet_NaN()), r_dual(std::numeric_limits<T>::quiet_NaN()), iterations(0) {
        if (!use_gpu) {
            x = thrust::host_vector<T>(n);
            z = thrust::host_vector<T>(m);
            s = thrust::host_vector<T>(m);
        }
    }
};

template <typename T>
class Solver {
public:
    std::optional<DefaultProblemData<T>> data;
    std::optional<DefaultVariables<T>> variables;
    std::variant<CompositeCone<T>, CompositeConeGPU<T>, std::monostate> cones;
    std::optional<DefaultResiduals<T>> residuals;
    std::optional<AbstractKKTSystem<T>> kktsystem;
    std::optional<DefaultInfo<T>> info;
    std::optional<DefaultVariables<T>> step_lhs, step_rhs, prev_vars;
    std::optional<DefaultSolution<T>> solution;
    std::optional<bool> use_gpu;
    Settings<T> settings;
    TimerOutput timers;

    Solver(const Settings<T>& settings)
        : settings(settings), timers(TimerOutput()) {
        timers.add_section("setup!");
        timers.add_section("solve!");
        timers.reset("setup!");
        timers.reset("solve!");
    }

    Solver() : Solver(Settings<T>()) {}

    Solver(const std::map<std::string, T>& d) : Solver(Settings<T>(d)) {}
};

} // namespace Clarabel
