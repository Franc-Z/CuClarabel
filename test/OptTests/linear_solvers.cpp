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
#include <thrust/iterator/zip_function.h>

template <typename T>
std::tuple<thrust::device_vector<T>, thrust::device_vector<T>, thrust::device_vector<T>, thrust::device_vector<T>, std::vector<SupportedCone>> basic_QP_data() {
    // Implement the function to generate basic QP data
    // This is a placeholder implementation
    thrust::device_vector<T> P, c, A, b;
    std::vector<SupportedCone> cones;
    return std::make_tuple(P, c, A, b, cones);
}

template <typename T>
std::tuple<thrust::device_vector<T>, thrust::device_vector<T>, thrust::device_vector<T>, thrust::device_vector<T>, std::vector<SupportedCone>> basic_SOCP_data() {
    // Implement the function to generate basic SOCP data
    // This is a placeholder implementation
    thrust::device_vector<T> P, c, A, b;
    std::vector<SupportedCone> cones;
    return std::make_tuple(P, c, A, b, cones);
}

template <typename T>
std::tuple<thrust::device_vector<T>, thrust::device_vector<T>, thrust::device_vector<T>, thrust::device_vector<T>, std::vector<SupportedCone>> basic_SDP_data() {
    // Implement the function to generate basic SDP data
    // This is a placeholder implementation
    thrust::device_vector<T> P, c, A, b;
    std::vector<SupportedCone> cones;
    return std::make_tuple(P, c, A, b, cones);
}

template <typename T>
void test_linear_solve(const std::string& solver_type) {
    T tol = static_cast<T>(1e-3);

    Settings<T> settings;
    settings.direct_solve_method = solver_type;

    auto [P, c, A, b, cones] = basic_QP_data<T>();
    Solver<T> solver;
    solver.setup(P, c, A, b, cones, settings);
    auto solution = solver.solve();

    assert(solution.status == SOLVED);
    assert(std::abs(thrust::reduce(solution.x.begin(), solution.x.end(), static_cast<T>(0), thrust::plus<T>()) - static_cast<T>(0.3 + 0.7)) < tol);
    assert(std::abs(solution.obj_val - static_cast<T>(1.8800000298331538)) < tol);

    auto [P_socp, c_socp, A_socp, b_socp, cones_socp] = basic_SOCP_data<T>();
    solver.setup(P_socp, c_socp, A_socp, b_socp, cones_socp, settings);
    solution = solver.solve();

    assert(solution.status == SOLVED);
    assert(std::abs(thrust::reduce(solution.x.begin(), solution.x.end(), static_cast<T>(0), thrust::plus<T>()) - static_cast<T>(-0.5 + 0.435603 - 0.245459)) < tol);
    assert(std::abs(solution.obj_val - static_cast<T>(-8.4590e-01)) < tol);

    auto [P_sdp, c_sdp, A_sdp, b_sdp, cones_sdp] = basic_SDP_data<T>();
    solver.setup(P_sdp, c_sdp, A_sdp, b_sdp, cones_sdp, settings);
    solution = solver.solve();

    thrust::device_vector<T> refsol = {static_cast<T>(-3.0729833267361095), static_cast<T>(0.3696004167288786), static_cast<T>(-0.022226685581313674), static_cast<T>(0.31441213129613066), static_cast<T>(-0.026739700851545107), static_cast<T>(-0.016084530571308823)};

    assert(solution.status == SOLVED);
    assert(std::abs(thrust::reduce(solution.x.begin(), solution.x.end(), static_cast<T>(0), thrust::plus<T>()) - thrust::reduce(refsol.begin(), refsol.end(), static_cast<T>(0), thrust::plus<T>())) < tol);
    assert(std::abs(solution.obj_val - static_cast<T>(4.840076866013861)) < tol);
}

int main() {
    test_linear_solve<float>("qdldl");
    test_linear_solve<double>("qdldl");
    test_linear_solve<float>("cholmod");
    test_linear_solve<double>("cholmod");
    return 0;
}
