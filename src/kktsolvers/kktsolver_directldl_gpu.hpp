#include <cuda_runtime.h>
#include <cusparse.h>
#include <CUDSS/CudssSolver.h>
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cassert>

template <typename T>
class GPULDLKKTSolver {
public:
    // problem dimensions
    int m, n;

    // Left and right hand sides for solves
    std::vector<T> x;
    std::vector<T> b;

    // internal workspace for IR scheme
    // and static offsetting of KKT
    std::vector<T> work1;
    std::vector<T> work2;

    // KKT mapping from problem data to KKT
    FullDataMap mapcpu;
    GPUDataMap mapgpu;

    // the expected signs of D in KKT = LDL^T
    std::vector<int> Dsigns;

    // a vector for storing the Hs blocks
    // on the in the KKT matrix block diagonal
    std::vector<T> Hsblocks;

    // unpermuted KKT matrix
    SparseMatrixCSC<T, int> KKTcpu;
    AbstractCuSparseMatrix<T> KKTgpu;

    // settings just points back to the main solver settings.
    // Required since there is no separate LDL settings container
    Settings<T> settings;

    // the direct linear LDL solver
    AbstractGPUSolver<T> GPUsolver;

    // the diagonal regularizer currently applied
    T diagonal_regularizer;

    GPULDLKKTSolver(const SparseMatrixCSC<T>& P, const SparseMatrixCSC<T>& A, const CompositeCone<T>& cones, int m, int n, const Settings<T>& settings)
        : m(m), n(n), settings(settings) {
        // get a constructor for the LDL solver we should use,
        // and also the matrix shape it requires
        auto [kktshape, GPUsolverT] = _get_GPUsolver_config(settings);

        // construct a KKT matrix of the right shape
        auto [KKTcpu, mapcpu] = _assemble_full_kkt_matrix(P, A, cones, kktshape);
        KKTgpu = CuSparseMatrixCSR(KKTcpu);

        // update GPU map, should be removed later on
        mapgpu = GPUDataMap(P, A, cones, mapcpu);

        // disabled sparse expansion and preprocess a large second-order cone into multiple small cones
        int p = pdim(mapcpu.sparse_maps);
        assert(p == 0);

        // updates to the diagonal of KKT will be
        // assigned here before updating matrix entries
        int dim = m + n;

        // LHS/RHS/work for iterative refinement
        x.resize(dim);
        b.resize(dim);
        work1.resize(dim);
        work2.resize(dim);

        // the expected signs of D in LDL
        std::vector<int> Dsigns_cpu(dim);
        _fill_Dsigns(Dsigns_cpu, m, n, mapcpu); // This is run on CPU
        Dsigns = Dsigns_cpu;

        Hsblocks.resize(_allocate_kkt_Hsblocks(T, cones));

        diagonal_regularizer = T(0);

        // the indirect linear solver engine
        GPUsolver = GPUsolverT(KKTgpu, x, b);
    }

    void update(const CompositeConeGPU<T>& cones) {
        // the internal GPUsolver is type unstable, so multiple
        // calls to the GPUsolvers will be very slow if called
        // directly. Grab it here and then call an inner function
        // so that the GPUsolver has concrete type
        _update_inner(cones);
    }

    void set_rhs(const std::vector<T>& rhsx, const std::vector<T>& rhsz) {
        std::copy(rhsx.begin(), rhsz.end(), b.begin());
        std::copy(rhsz.begin(), rhsz.end(), b.begin() + n);
        cudaDeviceSynchronize();
    }

    void get_lhs(std::vector<T>& lhsx, std::vector<T>& lhsz) {
        std::copy(x.begin(), x.begin() + n, lhsx.begin());
        std::copy(x.begin() + n, x.end(), lhsz.begin());
        cudaDeviceSynchronize();
    }

    bool solve(std::vector<T>& lhsx, std::vector<T>& lhsz) {
        solve(GPUsolver, x, b);

        bool is_success = false;
        if (settings.iterative_refinement_enable) {
            // IR reports success based on finite normed residual
            is_success = _iterative_refinement();
        } else {
            // otherwise must directly verify finite values
            is_success = std::all_of(x.begin(), x.end(), [](T val) { return std::isfinite(val); });
        }

        if (is_success) {
            get_lhs(lhsx, lhsz);
        }

        return is_success;
    }

private:
    void _update_inner(const CompositeConeGPU<T>& cones) {
        // real implementation is here, and now GPUsolver
        // will be compiled to something concrete.

        // Set the elements the W^tW blocks in the KKT matrix.
        // get_Hs!(cones, Hsblocks, false)
        get_Hs(cones, Hsblocks);

        std::transform(Hsblocks.begin(), Hsblocks.end(), Hsblocks.begin(), [](T val) { return -val; });
        _update_values(GPUsolver, KKTgpu, mapgpu.Hsblocks, Hsblocks);

        _regularize_and_refactor();
    }

    void _regularize_and_refactor() {
        if (settings.static_regularization_enable) {
            // hold a copy of the true KKT diagonal
            std::copy(KKTgpu.nzVal.begin() + mapgpu.diag_full[0], KKTgpu.nzVal.begin() + mapgpu.diag_full[1], work1.begin());
            T ϵ = _compute_regularizer(work1, settings);

            // compute an offset version, accounting for signs
            std::transform(work1.begin(), work1.end(), Dsigns.begin(), work2.begin(), [ϵ](T diag, int sign) { return diag + sign * ϵ; });

            // overwrite the diagonal of KKT and within the GPUsolver
            _update_diag_values_KKT(KKTgpu, mapgpu.diag_full, work2);

            // remember the value we used. Not needed,
            // but possibly useful for debugging
            diagonal_regularizer = ϵ;
        }

        bool is_success = refactor(GPUsolver);

        if (settings.static_regularization_enable) {
            // put our internal copy of the KKT matrix back the way
            // it was. Not necessary to fix the GPUsolver copy because
            // this is only needed for our post-factorization IR scheme
            _update_diag_values_KKT(KKTgpu, mapgpu.diag_full, work1);
        }

        assert(is_success);
    }

    bool _iterative_refinement() {
        T normb = *std::max_element(b.begin(), b.end(), [](T a, T b) { return std::abs(a) < std::abs(b); });

        // compute the initial error
        T norme = _get_refine_error(work1, b, KKTgpu, x);
        if (!std::isfinite(norme)) return false;

        for (int i = 0; i < settings.iterative_refinement_max_iter; ++i) {
            if (norme <= settings.iterative_refinement_abstol + settings.iterative_refinement_reltol * normb) {
                // within tolerance, or failed. Exit
                break;
            }
            T lastnorme = norme;

            // make a refinement and continue
            solve(GPUsolver, work2, work1);

            // prospective solution is x + dx. Use dx space to
            // hold it for a check before applying to x
            std::transform(work2.begin(), work2.end(), x.begin(), work2.begin(), std::plus<T>());
            cudaDeviceSynchronize();
            norme = _get_refine_error(work1, b, KKTgpu, work2);
            if (!std::isfinite(norme)) return false;

            T improved_ratio = lastnorme / norme;
            if (improved_ratio < settings.iterative_refinement_stop_ratio) {
                // insufficient improvement. Exit
                if (improved_ratio > T(1)) {
                    std::swap(x, work2);
                }
                break;
            }
            std::swap(x, work2);
        }

        return true;
    }

    T _get_refine_error(std::vector<T>& e, const std::vector<T>& b, const AbstractCuSparseMatrix<T>& KKT, const std::vector<T>& ξ) {
        // computes e = b - Kξ, overwriting the first argument
        // and returning its norm

        mul(e, KKT, ξ); // e = b - Kξ
        std::transform(b.begin(), b.end(), e.begin(), e.begin(), std::minus<T>());
        cudaDeviceSynchronize();
        T norme = *std::max_element(e.begin(), e.end(), [](T a, T b) { return std::abs(a) < std::abs(b); });

        return norme;
    }
};

template <typename T>
GPULDLKKTSolver<T> make_GPULDLKKTSolver(const SparseMatrixCSC<T>& P, const SparseMatrixCSC<T>& A, const CompositeCone<T>& cones, int m, int n, const Settings<T>& settings) {
    return GPULDLKKTSolver<T>(P, A, cones, m, n, settings);
}

template <typename T>
AbstractGPUSolver<T> _get_GPUsolver_type(const std::string& s) {
    try {
        return GPUSolversDict.at(s);
    } catch (const std::out_of_range&) {
        throw std::runtime_error("Unsupported gpu linear solver: " + s);
    }
}

template <typename T>
std::pair<std::string, AbstractGPUSolver<T>> _get_GPUsolver_config(const Settings<T>& settings) {
    // which LDL solver should I use?
    auto GPUsolverT = _get_GPUsolver_type<T>(settings.direct_solve_method);

    // does it want a :full KKT matrix?
    std::string kktshape = required_matrix_shape(typeid(GPUsolverT));
    assert(kktshape == "full");

    return {kktshape, GPUsolverT};
}

template <typename T, typename Ti>
void _update_values(AbstractGPUSolver<T>& GPUsolver, AbstractSparseMatrix<T>& KKT, const std::vector<Ti>& index, const std::vector<T>& values) {
    // Update values in the KKT matrix K
    for (size_t i = 0; i < index.size(); ++i) {
        KKT.nzVal[index[i]] = values[i];
    }
}

template <typename T, typename Ti>
void _update_diag_values_KKT(AbstractCuSparseMatrix<T>& KKT, const std::vector<Ti>& index, const std::vector<T>& values) {
    // Update values in the KKT matrix K
    std::copy(values.begin(), values.end(), KKT.nzVal.begin() + index[0]);
}
