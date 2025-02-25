#include <MathOptInterface.h>
#include <SparseArrays.h>
#include "Clarabel.h"

using namespace MathOptInterface;
using namespace SparseArrays;
using namespace Clarabel;

class Optimizer {
public:
    using T = double; // Assuming T is double, change if necessary

    Module solver_module;
    std::unique_ptr<Clarabel::AbstractSolver<T>> solver;
    Clarabel::Settings<T> solver_settings;
    std::unique_ptr<Clarabel::DefaultInfo<T>> solver_info;
    std::unique_ptr<Clarabel::DefaultSolution<T>> solver_solution;
    std::optional<DefaultInt> solver_nvars;
    bool use_quad_obj;
    MOI::OptimizationSense sense;
    T objconstant;
    std::unordered_map<DefaultInt, std::pair<DefaultInt, DefaultInt>> rowranges;

    Optimizer(Module solver_module = Clarabel, std::unordered_map<std::string, T> user_settings = {}) 
        : solver_module(solver_module), solver(nullptr), solver_settings(Clarabel::Settings<T>()), 
          solver_info(nullptr), solver_solution(nullptr), solver_nvars(std::nullopt), use_quad_obj(true), 
          sense(MOI::MIN_SENSE), objconstant(0), rowranges() {
        for (const auto& [key, value] : user_settings) {
            MOI::set(*this, MOI::RawOptimizerAttribute(key), value);
        }
    }

    void empty() {
        solver = nullptr;
        solver_settings = solver_settings;
        solver_info = nullptr;
        solver_solution = nullptr;
        solver_nvars = std::nullopt;
        sense = MOI::MIN_SENSE;
        objconstant = 0;
        rowranges.clear();
    }

    bool is_empty() const {
        return solver == nullptr;
    }

    void optimize() {
        if (solver_module == Clarabel) {
            solver_solution = std::make_unique<Clarabel::DefaultSolution<T>>(Clarabel::solve(*solver));
        } else {
            solver_solution = std::make_unique<Clarabel::DefaultSolution<T>>(solver_module.solve(*solver));
        }
        solver_info = std::make_unique<Clarabel::DefaultInfo<T>>(solver_module.get_info(*solver));
    }

    void show(std::ostream& os) const {
        std::string myname = MOI::get(*this, MOI::SolverName());
        if (solver == nullptr) {
            os << "Empty " << myname << " - Optimizer";
        } else {
            os << myname << " - Optimizer\n";
            os << " : Has results: " << (solver_solution == nullptr) << "\n";
            os << " : Objective constant: " << objconstant << "\n";
            os << " : Sense: " << sense << "\n";
            os << " : Precision: " << typeid(T).name() << "\n";

            if (solver_solution != nullptr) {
                os << " : Problem status: " << MOI::get(*this, MOI::RawStatusString()) << "\n";
                T value = round(MOI::get(*this, MOI::ObjectiveValue()), 3);
                os << " : Optimal objective: " << value << "\n";
                os << " : Iterations: " << MOI::get(*this, MOI::BarrierIterations()) << "\n";
                T solvetime = round(solver_info->solve_time * 1000, 2);
                os << " : Solve time: " << solvetime << "ms\n";
            }
        }
    }

    // Add other methods and attributes as needed
};

// Add other necessary functions and classes

