#include <iostream>
#include <vector>
#include <tuple>
#include <stdexcept>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/zip_function.h>
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>
#include <thrust/transform_reduce.h>
#include <thrust/transform_scan.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>

namespace Clarabel {

template <typename T>
using MatrixProblemDataUpdate = std::variant<std::monostate, thrust::device_vector<T>, thrust::host_vector<T>, std::vector<T>, std::tuple<std::vector<int>, std::vector<T>>>;

template <typename T>
using VectorProblemDataUpdate = std::variant<std::monostate, thrust::device_vector<T>, thrust::host_vector<T>, std::vector<T>, std::tuple<std::vector<int>, std::vector<T>>>;

template <typename T>
class Solver {
public:
    // Solver class definition
    // ...
    void update_data(const MatrixProblemDataUpdate<T>& P, const VectorProblemDataUpdate<T>& q, const MatrixProblemDataUpdate<T>& A, const VectorProblemDataUpdate<T>& b) {
        update_P(P);
        update_q(q);
        update_A(A);
        update_b(b);
    }

private:
    // Solver class members
    // ...
    void update_P(const MatrixProblemDataUpdate<T>& data) {
        if (std::holds_alternative<std::monostate>(data)) return;
        check_update_allowed();
        auto d = data.equilibration.d;
        update_matrix(data, data.P, d, d);
        kkt_update_P(kktsystem, data.P);
    }

    void update_A(const MatrixProblemDataUpdate<T>& data) {
        if (std::holds_alternative<std::monostate>(data)) return;
        check_update_allowed();
        auto d = data.equilibration.d;
        auto e = data.equilibration.e;
        update_matrix(data, data.A, e, d);
        kkt_update_A(kktsystem, data.A);
    }

    void update_q(const VectorProblemDataUpdate<T>& data) {
        if (std::holds_alternative<std::monostate>(data)) return;
        check_update_allowed();
        auto d = data.equilibration.d;
        auto dinv = data.equilibration.dinv;
        update_vector(data, data.q, d);
        data_clear_normq(data);
    }

    void update_b(const VectorProblemDataUpdate<T>& data) {
        if (std::holds_alternative<std::monostate>(data)) return;
        check_update_allowed();
        auto e = data.equilibration.e;
        auto einv = data.equilibration.einv;
        update_vector(data, data.b, e);
        data_clear_normb(data);
    }

    void check_update_allowed() {
        if (settings.presolve_enable || settings.chordal_decomposition_enable || data.presolver || data.chordal_info) {
            throw std::runtime_error("Disable presolve and chordal decomposition to allow data updates.");
        }
    }

    void update_matrix(const thrust::device_vector<T>& data, thrust::device_vector<T>& M, const thrust::device_vector<T>& lscale, const thrust::device_vector<T>& rscale) {
        if (data.size() != M.size()) throw std::runtime_error("Input must match length of original data.");
        thrust::transform(thrust::device, data.begin(), data.end(), M.begin(), thrust::multiplies<T>());
    }

    void update_matrix(const std::tuple<std::vector<int>, std::vector<T>>& data, thrust::device_vector<T>& M, const thrust::device_vector<T>& lscale, const thrust::device_vector<T>& rscale) {
        auto indices = std::get<0>(data);
        auto values = std::get<1>(data);
        for (size_t i = 0; i < indices.size(); ++i) {
            int idx = indices[i];
            if (idx < 0 || idx >= M.size()) throw std::runtime_error("Input must match sparsity pattern of original data.");
            M[idx] = lscale[idx] * rscale[idx] * values[i];
        }
    }

    void update_vector(const thrust::device_vector<T>& data, thrust::device_vector<T>& v, const thrust::device_vector<T>& scale) {
        if (data.size() != v.size()) throw std::runtime_error("Input must match length of original data.");
        thrust::transform(thrust::device, data.begin(), data.end(), scale.begin(), v.begin(), thrust::multiplies<T>());
    }

    void update_vector(const std::tuple<std::vector<int>, std::vector<T>>& data, thrust::device_vector<T>& v, const thrust::device_vector<T>& scale) {
        auto indices = std::get<0>(data);
        auto values = std::get<1>(data);
        for (size_t i = 0; i < indices.size(); ++i) {
            int idx = indices[i];
            if (idx < 0 || idx >= v.size()) throw std::runtime_error("Input must match length of original data.");
            v[idx] = values[i] * scale[idx];
        }
    }

    // Other private members and methods
    // ...
};

} // namespace Clarabel
