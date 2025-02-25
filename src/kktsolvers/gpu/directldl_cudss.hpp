#include <cuda_runtime.h>
#include <cusparse.h>
#include <CUDSS/CudssSolver.h>
#include <vector>
#include <map>

class CUDSSDirectLDLSolver {
public:
    using T = float; // Assuming T is float, change if necessary

    CUDSSDirectLDLSolver(const std::vector<int>& KKT, std::vector<T>& x, std::vector<T>& b)
        : KKTgpu(KKT), x(x), b(b) {
        int dim = KKT.size(); // Assuming KKT is a square matrix stored in a 1D vector

        // make a logical factorization to fix memory allocations
        // "S" denotes real symmetric and 'U' denotes the upper triangular

        cudssSolver = CUDSS::CudssSolver<T>(KKTgpu, "S", 'F');

        cudss("analysis", cudssSolver, x, b);
        cudss("factorization", cudssSolver, x, b);
    }

    void refactor() {
        // Update the KKT matrix in the cudss solver
        cudss_set(cudssSolver.matrix, KKTgpu);

        // Refactorization
        cudss("factorization", cudssSolver, x, b);

        // YC: should be corrected later on 
        // return all(isfinite, cudss_get(ldlsolver.cudssSolver.data,"diag"))
    }

    void solve(std::vector<T>& x, const std::vector<T>& b) {
        // solve on GPU
        ldiv(x, cudssSolver, b);
    }

private:
    std::vector<int> KKTgpu;
    CUDSS::CudssSolver<T> cudssSolver;
    std::vector<T> x;
    std::vector<T> b;
};

std::map<std::string, CUDSSDirectLDLSolver> GPUSolversDict = { {"cudss", CUDSSDirectLDLSolver()} };

std::string required_matrix_shape(const std::type_info& type) {
    if (type == typeid(CUDSSDirectLDLSolver)) {
        return "full";
    }
    return "";
}
