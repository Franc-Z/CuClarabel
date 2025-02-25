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
#include <random>
#include "Clarabel.hpp"
#include "coneops_socone_gpu.hpp"

template <typename T>
std::tuple<thrust::device_vector<T>, thrust::device_vector<T>, thrust::device_vector<T>, thrust::device_vector<T>, std::vector<SupportedCone>> basic_SOCP_data() {
    std::mt19937 rng(242713);
    int n = 3;
    thrust::host_vector<T> P_host(n * n);
    std::generate(P_host.begin(), P_host.end(), [&]() { return static_cast<T>(rng() / static_cast<T>(rng.max())); });
    thrust::device_vector<T> P = P_host;
    thrust::device_vector<T> A(n * n, static_cast<T>(1));
    thrust::device_vector<T> A1 = A;
    thrust::transform(A1.begin(), A1.end(), A1.begin(), [](T val) { return val * 2; });
    thrust::device_vector<T> c = {static_cast<T>(0.1), static_cast<T>(-2.0), static_cast<T>(1.0)};
    thrust::device_vector<T> b1(6, static_cast<T>(1));
    std::vector<SupportedCone> cones = {NonnegativeConeT(3), NonnegativeConeT(3)};

    thrust::device_vector<T> A2(n * n, static_cast<T>(1));
    thrust::device_vector<T> b2 = {static_cast<T>(0), static_cast<T>(0), static_cast<T>(0)};
    thrust::device_vector<T> A_combined(A1.size() + A2.size());
    thrust::copy(A1.begin(), A1.end(), A_combined.begin());
    thrust::copy(A2.begin(), A2.end(), A_combined.begin() + A1.size());
    thrust::device_vector<T> b_combined(b1.size() + b2.size());
    thrust::copy(b1.begin(), b1.end(), b_combined.begin());
    thrust::copy(b2.begin(), b2.end(), b_combined.begin() + b1.size());
    cones.push_back(SecondOrderConeT(3));

    return std::make_tuple(P, c, A_combined, b_combined, cones);
}

template <typename T>
void test_basic_SOCP() {
    T tol = static_cast<T>(1e-3);

    auto [P, c, A, b, cones] = basic_SOCP_data<T>();
    Clarabel::Solver<T> solver;
    solver.setup(P, c, A, b, cones);
    auto solution = solver.solve();

    assert(solution.status == SOLVED);
    assert(std::abs(thrust::reduce(solution.x.begin(), solution.x.end(), static_cast<T>(0), thrust::plus<T>()) - static_cast<T>(-0.5 + 0.435603 - 0.245459)) < tol);
    assert(std::abs(solution.obj_val - static_cast<T>(-8.4590e-01)) < tol);
    assert(std::abs(solution.obj_val_dual - static_cast<T>(-8.4590e-01)) < tol);

    b[7] = static_cast<T>(-10);
    solver.setup(P, c, A, b, cones);
    solution = solver.solve();

    assert(solution.status == PRIMAL_INFEASIBLE);
    assert(std::isnan(solution.obj_val));
    assert(std::isnan(solution.obj_val_dual));
}

int main() {
    test_basic_SOCP<float>();
    test_basic_SOCP<double>();
    return 0;
}
