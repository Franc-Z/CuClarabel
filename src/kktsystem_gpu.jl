#include <vector>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/sequence.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/zip_iterator.h>

template <typename T>
class DefaultKKTSystemGPU {
public:
    // the KKT system solver
    AbstractKKTSolver<T> kktsolver;

    // solution vector for constant part of KKT solves
    thrust::device_vector<T> x1;
    thrust::device_vector<T> z1;

    // solution vector for general KKT solves
    thrust::device_vector<T> x2;
    thrust::device_vector<T> z2;

    // work vectors for assembling/disassembling vectors
    thrust::device_vector<T> workx;
    thrust::device_vector<T> workz;
    thrust::device_vector<T> work_conic;

    // temporary GPU vector for the conic part
    thrust::device_vector<T> workx2;

    DefaultKKTSystemGPU(DefaultProblemData<T>& data, CompositeCone<T>& cones, Settings<T>& settings)
        : kktsolver(GPULDLKKTSolver<T>(data.P, data.A, cones, data.m, data.n, settings)),
          x1(data.n), z1(data.m), x2(data.n), z2(data.m),
          workx(data.n), workz(data.m), work_conic(data.m), workx2(data.n) {}

    bool kkt_update(DefaultProblemData<T>& data, CompositeConeGPU<T>& cones) {
        // update the linear solver with new cones
        bool is_success = kktsolver.update(cones);

        // bail if the factorization has failed
        if (!is_success) return is_success;

        // calculate KKT solution for constant terms
        is_success = _kkt_solve_constant_rhs(data);

        return is_success;
    }

    bool _kkt_solve_constant_rhs(DefaultProblemData<T>& data) {
        thrust::transform(data.q_gpu.begin(), data.q_gpu.end(), workx.begin(), thrust::negate<T>());

        kktsolver.set_rhs(workx, data.b_gpu);
        bool is_success = kktsolver.solve(x2, z2);

        return is_success;
    }

    bool kkt_solve_initial_point(DefaultVariables<T>& variables, DefaultProblemData<T>& data) {
        if (data.P.nonZeros() == 0) {
            // LP initialization
            // solve with [0;b] as a RHS to get (x,-s) initializers
            // zero out any sparse cone variables at end
            thrust::fill(workx.begin(), workx.end(), T(0));
            thrust::copy(data.b_gpu.begin(), data.b_gpu.end(), workz.begin());
            kktsolver.set_rhs(workx, workz);
            bool is_success = kktsolver.solve(variables.x, variables.s);
            thrust::transform(variables.s.begin(), variables.s.end(), variables.s.begin(), thrust::negate<T>());

            if (!is_success) return is_success;

            // solve with [-q;0] as a RHS to get z initializer
            // zero out any sparse cone variables at end
            thrust::transform(data.q_gpu.begin(), data.q_gpu.end(), workx.begin(), thrust::negate<T>());
            thrust::fill(workz.begin(), workz.end(), T(0));

            kktsolver.set_rhs(workx, workz);
            is_success = kktsolver.solve(thrust::nullopt, variables.z);
        } else {
            // QP initialization
            thrust::transform(data.q_gpu.begin(), data.q_gpu.end(), workx.begin(), thrust::negate<T>());
            thrust::copy(data.b_gpu.begin(), data.b_gpu.end(), workz.begin());

            kktsolver.set_rhs(workx, workz);
            bool is_success = kktsolver.solve(variables.x, variables.z);
            thrust::transform(variables.z.begin(), variables.z.end(), variables.s.begin(), thrust::negate<T>());
        }

        return is_success;
    }

    bool kkt_solve(DefaultVariables<T>& lhs, DefaultVariables<T>& rhs, DefaultProblemData<T>& data, DefaultVariables<T>& variables, CompositeConeGPU<T>& cones, const std::string& steptype) {
        // solve for (x1,z1)
        thrust::copy(rhs.x.begin(), rhs.x.end(), workx.begin());

        // compute the vector c in the step equation HₛΔz + Δs = -c,
        // with shortcut in affine case
        thrust::device_vector<T> Δs_const_term = work_conic;

        if (steptype == "affine") {
            thrust::copy(variables.s.begin(), variables.s.end(), Δs_const_term.begin());
        } else {
            // we can use the overall LHS output as additional workspace for the moment
            Δs_from_Δz_offset(cones, Δs_const_term, rhs.s, lhs.z, variables.z);
        }

        thrust::transform(Δs_const_term.begin(), Δs_const_term.end(), rhs.z.begin(), workz.begin(), thrust::minus<T>());
        cudaDeviceSynchronize();

        // this solves the variable part of reduced KKT system
        kktsolver.set_rhs(workx, workz);
        bool is_success = kktsolver.solve(x1, z1);

        if (!is_success) return false;

        // solve for Δτ
        thrust::transform(variables.x.begin(), variables.x.end(), thrust::make_constant_iterator(variables.τ), workx.begin(), thrust::divides<T>());
        cudaDeviceSynchronize();

        thrust::device_vector<T> workx2 = this->workx2;
        mul(workx2, data.P_gpu, x1);
        T tau_num = rhs.τ - rhs.κ / variables.τ + thrust::inner_product(data.q_gpu.begin(), data.q_gpu.end(), x1.begin(), T(0)) + thrust::inner_product(data.b_gpu.begin(), data.b_gpu.end(), z1.begin(), T(0)) + 2 * thrust::inner_product(workx.begin(), workx.end(), workx2.begin(), T(0));

        thrust::transform(workx.begin(), workx.end(), x2.begin(), workx.begin(), thrust::minus<T>());
        cudaDeviceSynchronize();

        T tau_den = variables.κ / variables.τ - thrust::inner_product(data.q_gpu.begin(), data.q_gpu.end(), x2.begin(), T(0)) - thrust::inner_product(data.b_gpu.begin(), data.b_gpu.end(), z2.begin(), T(0));
        mul(workx2, data.P_gpu, workx);
        T t1 = thrust::inner_product(workx.begin(), workx.end(), workx2.begin(), T(0));
        mul(workx2, data.P_gpu, x2);
        T t2 = thrust::inner_product(x2.begin(), x2.end(), workx2.begin(), T(0));
        tau_den += t1 - t2;

        // solve for (Δx,Δz)
        lhs.τ = tau_num / tau_den;
        thrust::transform(x1.begin(), x1.end(), thrust::make_constant_iterator(lhs.τ), x2.begin(), lhs.x.begin(), thrust::plus<T>());
        thrust::transform(z1.begin(), z1.end(), thrust::make_constant_iterator(lhs.τ), z2.begin(), lhs.z.begin(), thrust::plus<T>());
        cudaDeviceSynchronize();

        // solve for Δs
        mul_Hs(cones, lhs.s, lhs.z, workz);
        thrust::transform(lhs.s.begin(), lhs.s.end(), Δs_const_term.begin(), lhs.s.begin(), thrust::minus<T>());
        cudaDeviceSynchronize();

        // solve for Δκ
        lhs.κ = -(rhs.κ + variables.κ * lhs.τ) / variables.τ;

        // we don't check the validity of anything
        // after the KKT solve, so just return is_success
        // without further validation
        return is_success;
    }
};
