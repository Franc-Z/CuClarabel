#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <limits>
#include <memory>
#include <cassert>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
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
class Solver {
public:
    Solver() = default;

    void setup(const thrust::device_vector<T>& P, const thrust::device_vector<T>& c, const thrust::device_vector<T>& A, const thrust::device_vector<T>& b, const std::vector<SupportedCone>& cones, const Settings<T>& settings) {
        this->settings = settings;
        setup(P, c, A, b, cones);
    }

    void setup(const thrust::device_vector<T>& P, const thrust::device_vector<T>& c, const thrust::device_vector<T>& A, const thrust::device_vector<T>& b, const std::vector<SupportedCone>& cones) {
        settings_populate(settings, {});
        setup(P, c, A, b, cones);
    }

    void setup(const thrust::device_vector<T>& P, const thrust::device_vector<T>& c, const thrust::device_vector<T>& A, const thrust::device_vector<T>& b, const std::vector<SupportedCone>& cones) {
        // project against cones with overly specific type, e.g.
        // when all of the cones are NonnegativeConeT
        auto converted_cones = convert(cones);

        // sanity check problem dimensions
        check_dimensions(P, c, A, b, converted_cones);

        // make this first to create the timers
        info = DefaultInfo<T>();

        // GPU preprocess
        use_gpu = gpu_preprocess(settings);
        bool use_full = use_gpu; // default gpu mapping type

        // user facing results go here
        solution = DefaultSolution<T>(A.size(), b.size(), use_gpu);

        // presolve / chordal decomposition if needed,
        // then take an internal copy of the problem data
        data = DefaultProblemData<T>(P, c, A, b, converted_cones, settings);

        CompositeCone<T> cpucones(data.cones, use_full);
        cones = use_gpu ? CompositeConeGPU<T>(cpucones) : cpucones;

        if (data.m != cones.numel()) {
            throw std::runtime_error("DimensionMismatch");
        }

        variables = DefaultVariables<T>(data.n, data.m, use_gpu);
        residuals = DefaultResiduals<T>(data.n, data.m, use_gpu);

        // equilibrate problem data immediately on setup.
        // this prevents multiple equlibrations if solve!
        // is called more than once.
        data_equilibrate(data, cpucones, settings);

        if (use_gpu) {
            gpu_data_copy(data); // YC: copy data to GPU, should be optimized later
            kktsystem = DefaultKKTSystemGPU<T>(data, cpucones, settings);
        } else {
            kktsystem = DefaultKKTSystem<T>(data, cones, settings);
        }

        // work variables for assembling step direction LHS/RHS
        step_rhs = DefaultVariables<T>(data.n, data.m, use_gpu);
        step_lhs = DefaultVariables<T>(data.n, data.m, use_gpu);

        // a saved copy of the previous iterate
        prev_vars = DefaultVariables<T>(data.n, data.m, use_gpu);
    }

    DefaultSolution<T> solve() {
        // initialization needed for first loop pass
        int iter = 0;
        T σ = 1;
        T α = 0;
        T μ = std::numeric_limits<T>::max();

        // select functions depending on devices
        bool use_gpu = this->use_gpu;
        auto residual_update = use_gpu ? residuals_update_gpu : residuals_update;
        auto info_update = use_gpu ? info_update_gpu : info_update;

        // solver release info, solver config
        // problem dimensions, cone type etc
        print_banner(settings.verbose);
        info_print_configuration(info, settings, data, cones);
        info_print_status_header(info, settings);

        info_reset(info, timers);

        // initialize variables to some reasonable starting point
        solver_default_start();

        // main loop
        ScalingStrategy scaling = allows_primal_dual_scaling(cones) ? PrimalDual::ScalingStrategy : Dual::ScalingStrategy;

        while (true) {
            // update the residuals
            residual_update(residuals, variables, data);

            // calculate duality gap (scaled)
            μ = variables_calc_mu(variables, residuals, cones);

            // record scalar values from most recent iteration.
            // This captures μ at iteration zero.
            info_save_scalars(info, μ, α, σ, iter);

            // convergence check and printing
            info_update(info, data, variables, residuals, kktsystem, settings, timers);
            info_print_status(info, settings);
            bool isdone = info_check_termination(info, residuals, settings, iter);

            // check for termination due to slow progress and update strategy
            if (isdone) {
                auto [action, new_scaling] = strategy_checkpoint_insufficient_progress(scaling, use_gpu);
                if (action == NoUpdate || action == Fail) {
                    break;
                } else if (action == Update) {
                    continue;
                }
            }

            // update the scalings
            bool is_scaling_success = variables_scale_cones(variables, cones, μ, scaling);

            // check whether variables are interior points
            auto [action, new_scaling] = strategy_checkpoint_is_scaling_success(is_scaling_success, scaling);
            if (action == Fail) {
                break;
            }

            // increment counter here because we only count
            // iterations that produce a KKT update
            iter++;

            // Update the KKT system and the constant parts of its solution.
            // Keep track of the success of each step that calls KKT
            bool is_kkt_solve_success = kkt_update(kktsystem, data, cones);

            // calculate the affine step
            variables_affine_step_rhs(step_rhs, residuals, variables, cones);
            is_kkt_solve_success = is_kkt_solve_success && kkt_solve(kktsystem, step_lhs, step_rhs, data, variables, cones, Affine);

            // combined step only on affine step success
            if (is_kkt_solve_success) {
                // calculate step length and centering parameter
                α = solver_get_step_length(Affine, scaling, use_gpu);
                σ = calc_centering_parameter(α);

                // make a reduced Mehrotra correction in the first iteration
                // to accommodate badly centred starting points
                T m = iter > 1 ? 1 : α;

                // calculate the combined step and length
                variables_combined_step_rhs(step_rhs, residuals, variables, cones, step_lhs, σ, μ, m);
                is_kkt_solve_success = kkt_solve(kktsystem, step_lhs, step_rhs, data, variables, cones, Combined);
            }

            // check for numerical failure and update strategy
            auto [action, new_scaling] = strategy_checkpoint_numerical_error(is_kkt_solve_success, scaling);
            if (action == NoUpdate) {
                // just keep going
            } else if (action == Update) {
                α = 0;
                continue;
            } else if (action == Fail) {
                α = 0;
                break;
            }

            // compute final step length and update the current iterate
            α = solver_get_step_length(Combined, scaling, use_gpu);

            // check for undersized step and update strategy
            auto [action, new_scaling] = strategy_checkpoint_small_step(α, scaling);
            if (action == NoUpdate) {
                // just keep going
            } else if (action == Update) {
                α = 0;
                continue;
            } else if (action == Fail) {
                α = 0;
                break;
            }

            // Copy previous iterate in case the next one is a dud
            info_save_prev_iterate(info, variables, prev_vars, use_gpu);

            variables_add_step(variables, step_lhs, α, use_gpu);
        }

        // Check we if actually took a final step. If not, we need
        // to recapture the scalars and print one last line
        if (α == 0) {
            info_save_scalars(info, μ, α, σ, iter);
            info_print_status(info, settings);
        }

        // check for "almost" convergence checks and then extract solution
        info_post_process(info, residuals, settings);
        solution_post_process(solution, data, variables, info, settings, use_gpu);

        // halt timers
        info_finalize(info, timers);
        solution_finalize(solution, info);

        info_print_footer(info, settings);

        return solution;
    }

private:
    DefaultInfo<T> info;
    DefaultSolution<T> solution;
    DefaultProblemData<T> data;
    CompositeCone<T> cones;
    DefaultVariables<T> variables;
    DefaultResiduals<T> residuals;
    DefaultVariables<T> step_rhs;
    DefaultVariables<T> step_lhs;
    DefaultVariables<T> prev_vars;
    DefaultKKTSystem<T> kktsystem;
    Settings<T> settings;
    bool use_gpu;

    void solver_default_start() {
        if (is_symmetric(cones)) {
            set_identity_scaling(cones);
            kkt_update(kktsystem, data, cones);
            kkt_solve_initial_point(kktsystem, variables, data);
            variables_symmetric_initialization(variables, cones);
        } else {
            variables_unit_initialization(variables, cones);
        }
    }

    T solver_get_step_length(StepType steptype, ScalingStrategy scaling, bool use_gpu) {
        T α = variables_calc_step_length(variables, step_lhs, cones, settings, steptype);

        if (!is_symmetric(cones) && steptype == Combined && scaling == Dual::ScalingStrategy) {
            T αinit = α;
            α = solver_backtrack_step_to_barrier(αinit);
        }

        return α;
    }

    T solver_backtrack_step_to_barrier(T αinit) {
        T step = settings.linesearch_backtrack_step;
        T α = αinit;

        for (int j = 0; j < 1; ++j) {
            T barrier = variables_barrier(variables, step_lhs, α, cones);
            if (barrier < 1) {
                return α;
            } else {
                α = step * α; // backtrack line search
            }
        }

        return α;
    }

    T calc_centering_parameter(T α) {
        return std::pow(1 - α, 3);
    }

    std::pair<StrategyCheckpoint, ScalingStrategy> strategy_checkpoint_insufficient_progress(ScalingStrategy scaling, bool use_gpu) {
        if (info.status != INSUFFICIENT_PROGRESS) {
            return {NoUpdate, scaling};
        } else {
            info_reset_to_prev_iterate(info, variables, prev_vars, use_gpu);

            if (!is_symmetric(cones) && scaling == PrimalDual::ScalingStrategy) {
                info.status = UNSOLVED;
                return {Update, Dual::ScalingStrategy};
            } else {
                return {Fail, scaling};
            }
        }
    }

    std::pair<StrategyCheckpoint, ScalingStrategy> strategy_checkpoint_numerical_error(bool is_kkt_solve_success, ScalingStrategy scaling) {
        if (is_kkt_solve_success) {
            return {NoUpdate, scaling};
        }

        if (!is_symmetric(cones) && scaling == PrimalDual::ScalingStrategy) {
            return {Update, Dual::ScalingStrategy};
        } else {
            info.status = NUMERICAL_ERROR;
            return {Fail, scaling};
        }
    }

    std::pair<StrategyCheckpoint, ScalingStrategy> strategy_checkpoint_small_step(T α, ScalingStrategy scaling) {
        if (!is_symmetric(cones) && scaling == PrimalDual::ScalingStrategy && α < settings.min_switch_step_length) {
            return {Update, Dual::ScalingStrategy};
        } else if (α <= std::max(static_cast<T>(0), settings.min_terminate_step_length)) {
            info.status = INSUFFICIENT_PROGRESS;
            return {Fail, scaling};
        } else {
            return {NoUpdate, scaling};
        }
    }

    std::pair<StrategyCheckpoint, ScalingStrategy> strategy_checkpoint_is_scaling_success(bool is_scaling_success, ScalingStrategy scaling) {
        if (is_scaling_success) {
            return {NoUpdate, scaling};
        } else {
            info.status = NUMERICAL_ERROR;
            return {Fail, scaling};
        }
    }
};

template <typename T>
bool gpu_preprocess(Settings<T>& settings) {
    if (settings.direct_solve_method == "cudssmixed") {
        assert(std::is_same<T, double>::value);
        settings.static_regularization_constant = std::sqrt(std::numeric_limits<float>::epsilon());
    }

    return settings.direct_kkt_solver && (std::find(gpu_solver_list.begin(), gpu_solver_list.end(), settings.direct_solve_method) != gpu_solver_list.end());
}

template <typename T>
void gpu_data_copy(DefaultProblemData<T>& data) {
    data.P_gpu = CuSparseMatrixCSR(data.P);
    data.q_gpu = thrust::device_vector<T>(data.q.begin(), data.q.end());
    data.A_gpu = CuSparseMatrixCSR(data.A);
    data.At_gpu = CuSparseMatrixCSR(data.A.transpose());
    data.b_gpu = thrust::device_vector<T>(data.b.begin(), data.b.end());
}
