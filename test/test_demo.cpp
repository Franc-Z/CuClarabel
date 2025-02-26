#include <iostream>
#include <vector>
#include <cassert>
#include "../src/Clarabel.hpp"

void test_solver() {
    std::vector<double> P = {1.0, 0.0, 0.0, 1.0};
    std::vector<double> A = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> cones = {1.0, 2.0};
    std::vector<double> settings = {0.1, 0.01};

    Clarabel::Solver<double> solver(2, 2, P, A, cones, settings);
    solver.solve();

    // Add assertions to verify the correctness of the solver
    // Example: assert(solver.get_result() == expected_result);
}

void test_settings() {
    Clarabel::Settings<double> settings;
    settings.set_parameter("tolerance", 1e-6);
    assert(settings.get_parameter("tolerance") == 1e-6);
}

void test_composite_cone() {
    std::vector<double> cones = {1.0, 2.0, 3.0};
    Clarabel::CompositeCone<double> composite_cone(cones);

    // Add assertions to verify the correctness of the composite cone
    // Example: assert(composite_cone.get_cones() == expected_cones);
}

int main() {
    test_solver();
    test_settings();
    test_composite_cone();

    std::cout << "All tests passed!" << std::endl;
    return 0;
}
