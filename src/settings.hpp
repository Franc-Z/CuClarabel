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
class Settings {
public:
    // User definable settings
    uint32_t max_iter = 200;
    double time_limit = std::numeric_limits<double>::infinity();
    bool verbose = true;
    T max_step_fraction = static_cast<T>(0.99);

    // Full accuracy solution tolerances
    T tol_gap_abs = static_cast<T>(1e-8);
    T tol_gap_rel = static_cast<T>(1e-8);
    T tol_feas = static_cast<T>(1e-8);
    T tol_infeas_abs = static_cast<T>(1e-8);
    T tol_infeas_rel = static_cast<T>(1e-8);
    T tol_ktratio = static_cast<T>(1e-6);

    // Reduced accuracy solution tolerances
    T reduced_tol_gap_abs = static_cast<T>(5e-5);
    T reduced_tol_gap_rel = static_cast<T>(5e-5);
    T reduced_tol_feas = static_cast<T>(1e-4);
    T reduced_tol_infeas_abs = static_cast<T>(5e-5);
    T reduced_tol_infeas_rel = static_cast<T>(5e-5);
    T reduced_tol_ktratio = static_cast<T>(1e-4);

    // Data equilibration
    bool equilibrate_enable = true;
    uint32_t equilibrate_max_iter = 10;
    T equilibrate_min_scaling = static_cast<T>(1e-4);
    T equilibrate_max_scaling = static_cast<T>(1e+4);

    // Cones and line search parameters
    T linesearch_backtrack_step = static_cast<T>(0.8);
    T min_switch_step_length = static_cast<T>(1e-2);
    T min_terminate_step_length = static_cast<T>(1e-4);

    // Direct linear solver package to use
    bool direct_kkt_solver = true; // Indirect not yet supported
    std::string direct_solve_method = "qdldl";

    // Static regularization parameters
    bool static_regularization_enable = true;
    T static_regularization_constant = static_cast<T>(1e-8);
    T static_regularization_proportional = std::numeric_limits<T>::epsilon() * std::numeric_limits<T>::epsilon();

    // Dynamic regularization parameters
    bool dynamic_regularization_enable = true;
    T dynamic_regularization_eps = static_cast<T>(1e-13);
    T dynamic_regularization_delta = static_cast<T>(2e-7);

    // Iterative refinement
    bool iterative_refinement_enable = true;
    T iterative_refinement_reltol = static_cast<T>(1e-12);
    T iterative_refinement_abstol = static_cast<T>(1e-12);
    int iterative_refinement_max_iter = 10;
    T iterative_refinement_stop_ratio = static_cast<T>(5);

    // Preprocessing
    bool presolve_enable = true;

    // Chordal decomposition
    bool chordal_decomposition_enable = true;
    std::string chordal_decomposition_merge_method = "clique_graph";
    bool chordal_decomposition_compact = true;
    bool chordal_decomposition_complete_dual = true;

    // Centrality check
    T neighborhood = static_cast<T>(1e-6);

    // Device: can be "cpu" or "gpu"
    std::string device = "cpu";

    Settings() = default;

    Settings(const std::map<std::string, T>& d) {
        populate(d);
    }

    void populate(const std::map<std::string, T>& d) {
        for (const auto& [key, val] : d) {
            if (key == "max_iter") max_iter = static_cast<uint32_t>(val);
            else if (key == "time_limit") time_limit = static_cast<double>(val);
            else if (key == "verbose") verbose = static_cast<bool>(val);
            else if (key == "max_step_fraction") max_step_fraction = val;
            else if (key == "tol_gap_abs") tol_gap_abs = val;
            else if (key == "tol_gap_rel") tol_gap_rel = val;
            else if (key == "tol_feas") tol_feas = val;
            else if (key == "tol_infeas_abs") tol_infeas_abs = val;
            else if (key == "tol_infeas_rel") tol_infeas_rel = val;
            else if (key == "tol_ktratio") tol_ktratio = val;
            else if (key == "reduced_tol_gap_abs") reduced_tol_gap_abs = val;
            else if (key == "reduced_tol_gap_rel") reduced_tol_gap_rel = val;
            else if (key == "reduced_tol_feas") reduced_tol_feas = val;
            else if (key == "reduced_tol_infeas_abs") reduced_tol_infeas_abs = val;
            else if (key == "reduced_tol_infeas_rel") reduced_tol_infeas_rel = val;
            else if (key == "reduced_tol_ktratio") reduced_tol_ktratio = val;
            else if (key == "equilibrate_enable") equilibrate_enable = static_cast<bool>(val);
            else if (key == "equilibrate_max_iter") equilibrate_max_iter = static_cast<uint32_t>(val);
            else if (key == "equilibrate_min_scaling") equilibrate_min_scaling = val;
            else if (key == "equilibrate_max_scaling") equilibrate_max_scaling = val;
            else if (key == "linesearch_backtrack_step") linesearch_backtrack_step = val;
            else if (key == "min_switch_step_length") min_switch_step_length = val;
            else if (key == "min_terminate_step_length") min_terminate_step_length = val;
            else if (key == "direct_kkt_solver") direct_kkt_solver = static_cast<bool>(val);
            else if (key == "direct_solve_method") direct_solve_method = static_cast<std::string>(val);
            else if (key == "static_regularization_enable") static_regularization_enable = static_cast<bool>(val);
            else if (key == "static_regularization_constant") static_regularization_constant = val;
            else if (key == "static_regularization_proportional") static_regularization_proportional = val;
            else if (key == "dynamic_regularization_enable") dynamic_regularization_enable = static_cast<bool>(val);
            else if (key == "dynamic_regularization_eps") dynamic_regularization_eps = val;
            else if (key == "dynamic_regularization_delta") dynamic_regularization_delta = val;
            else if (key == "iterative_refinement_enable") iterative_refinement_enable = static_cast<bool>(val);
            else if (key == "iterative_refinement_reltol") iterative_refinement_reltol = val;
            else if (key == "iterative_refinement_abstol") iterative_refinement_abstol = val;
            else if (key == "iterative_refinement_max_iter") iterative_refinement_max_iter = static_cast<int>(val);
            else if (key == "iterative_refinement_stop_ratio") iterative_refinement_stop_ratio = val;
            else if (key == "presolve_enable") presolve_enable = static_cast<bool>(val);
            else if (key == "chordal_decomposition_enable") chordal_decomposition_enable = static_cast<bool>(val);
            else if (key == "chordal_decomposition_merge_method") chordal_decomposition_merge_method = static_cast<std::string>(val);
            else if (key == "chordal_decomposition_compact") chordal_decomposition_compact = static_cast<bool>(val);
            else if (key == "chordal_decomposition_complete_dual") chordal_decomposition_complete_dual = static_cast<bool>(val);
            else if (key == "neighborhood") neighborhood = val;
            else if (key == "device") device = static_cast<std::string>(val);
        }
    }

    void show() const {
        std::cout << "Clarabel settings with Float precision: " << typeid(T).name() << "\n\n";

        std::vector<std::string> names = {
            "max_iter", "time_limit", "verbose", "max_step_fraction",
            "tol_gap_abs", "tol_gap_rel", "tol_feas", "tol_infeas_abs", "tol_infeas_rel", "tol_ktratio",
            "reduced_tol_gap_abs", "reduced_tol_gap_rel", "reduced_tol_feas", "reduced_tol_infeas_abs", "reduced_tol_infeas_rel", "reduced_tol_ktratio",
            "equilibrate_enable", "equilibrate_max_iter", "equilibrate_min_scaling", "equilibrate_max_scaling",
            "linesearch_backtrack_step", "min_switch_step_length", "min_terminate_step_length",
            "direct_kkt_solver", "direct_solve_method",
            "static_regularization_enable", "static_regularization_constant", "static_regularization_proportional",
            "dynamic_regularization_enable", "dynamic_regularization_eps", "dynamic_regularization_delta",
            "iterative_refinement_enable", "iterative_refinement_reltol", "iterative_refinement_abstol", "iterative_refinement_max_iter", "iterative_refinement_stop_ratio",
            "presolve_enable",
            "chordal_decomposition_enable", "chordal_decomposition_merge_method", "chordal_decomposition_compact", "chordal_decomposition_complete_dual",
            "neighborhood", "device"
        };

        std::vector<std::string> values = {
            std::to_string(max_iter), std::to_string(time_limit), std::to_string(verbose), std::to_string(max_step_fraction),
            std::to_string(tol_gap_abs), std::to_string(tol_gap_rel), std::to_string(tol_feas), std::to_string(tol_infeas_abs), std::to_string(tol_infeas_rel), std::to_string(tol_ktratio),
            std::to_string(reduced_tol_gap_abs), std::to_string(reduced_tol_gap_rel), std::to_string(reduced_tol_feas), std::to_string(reduced_tol_infeas_abs), std::to_string(reduced_tol_infeas_rel), std::to_string(reduced_tol_ktratio),
            std::to_string(equilibrate_enable), std::to_string(equilibrate_max_iter), std::to_string(equilibrate_min_scaling), std::to_string(equilibrate_max_scaling),
            std::to_string(linesearch_backtrack_step), std::to_string(min_switch_step_length), std::to_string(min_terminate_step_length),
            std::to_string(direct_kkt_solver), direct_solve_method,
            std::to_string(static_regularization_enable), std::to_string(static_regularization_constant), std::to_string(static_regularization_proportional),
            std::to_string(dynamic_regularization_enable), std::to_string(dynamic_regularization_eps), std::to_string(dynamic_regularization_delta),
            std::to_string(iterative_refinement_enable), std::to_string(iterative_refinement_reltol), std::to_string(iterative_refinement_abstol), std::to_string(iterative_refinement_max_iter), std::to_string(iterative_refinement_stop_ratio),
            std::to_string(presolve_enable),
            std::to_string(chordal_decomposition_enable), chordal_decomposition_merge_method, std::to_string(chordal_decomposition_compact), std::to_string(chordal_decomposition_complete_dual),
            std::to_string(neighborhood), device
        };

        std::vector<std::string> types = {
            typeid(max_iter).name(), typeid(time_limit).name(), typeid(verbose).name(), typeid(max_step_fraction).name(),
            typeid(tol_gap_abs).name(), typeid(tol_gap_rel).name(), typeid(tol_feas).name(), typeid(tol_infeas_abs).name(), typeid(tol_infeas_rel).name(), typeid(tol_ktratio).name(),
            typeid(reduced_tol_gap_abs).name(), typeid(reduced_tol_gap_rel).name(), typeid(reduced_tol_feas).name(), typeid(reduced_tol_infeas_abs).name(), typeid(reduced_tol_infeas_rel).name(), typeid(reduced_tol_ktratio).name(),
            typeid(equilibrate_enable).name(), typeid(equilibrate_max_iter).name(), typeid(equilibrate_min_scaling).name(), typeid(equilibrate_max_scaling).name(),
            typeid(linesearch_backtrack_step).name(), typeid(min_switch_step_length).name(), typeid(min_terminate_step_length).name(),
            typeid(direct_kkt_solver).name(), typeid(direct_solve_method).name(),
            typeid(static_regularization_enable).name(), typeid(static_regularization_constant).name(), typeid(static_regularization_proportional).name(),
            typeid(dynamic_regularization_enable).name(), typeid(dynamic_regularization_eps).name(), typeid(dynamic_regularization_delta).name(),
            typeid(iterative_refinement_enable).name(), typeid(iterative_refinement_reltol).name(), typeid(iterative_refinement_abstol).name(), typeid(iterative_refinement_max_iter).name(), typeid(iterative_refinement_stop_ratio).name(),
            typeid(presolve_enable).name(),
            typeid(chordal_decomposition_enable).name(), typeid(chordal_decomposition_merge_method).name(), typeid(chordal_decomposition_compact).name(), typeid(chordal_decomposition_complete_dual).name(),
            typeid(neighborhood).name(), typeid(device).name()
        };

        std::vector<std::string> titles = {"Setting", "DataType", "Value"};
        std::vector<std::string> dividers;

        for (size_t i = 0; i < titles.size(); ++i) {
            size_t len = std::max(static_cast<size_t>(8), std::max_element(values.begin(), values.end(), [](const std::string& a, const std::string& b) { return a.size() < b.size(); })->size());
            for (auto& value : values) {
                value.resize(len + 1, ' ');
            }
            titles[i].resize(len + 1, ' ');
            dividers.push_back(std::string(len + 2, '='));
        }

        std::cout << " ";
        for (const auto& title : titles) {
            std::cout << " " << title << " ";
        }
        std::cout << "\n ";

        for (const auto& divider : dividers) {
            std::cout << divider << " ";
        }
        std::cout << "\n";

        for (size_t row = 0; row < names.size(); ++row) {
            std::cout << " " << names[row] << " " << types[row] << " " << values[row] << "\n";
        }
        std::cout << "\n";
    }
};
