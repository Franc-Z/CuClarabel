#include <type_traits>
#include <vector>
#include <memory>
#include <iostream>
#include <unordered_map>
#include <functional>

// Forward declarations
template <typename T>
class PowerCone;

template <typename T>
class ExponentialCone;

template <typename T>
class AbstractCone;

template <typename T>
class CompositeCone;

// Type aliases for convenience
template <typename T>
using CONE3D_M3T_TYPE = std::vector<std::vector<T>>;

template <typename T>
using CONE3D_V3T_TYPE = std::vector<T>;

// Function to make cone type concrete
template <typename T>
struct make_conetype_concrete {
    using type = T;
};

template <typename T>
struct make_conetype_concrete<PowerCone<T>> {
    using type = PowerCone<T>;
};

template <typename T>
struct make_conetype_concrete<ExponentialCone<T>> {
    using type = ExponentialCone<T>;
};

// Function to dispatch on cone type
template <typename T, typename Func>
void conedispatch(const std::shared_ptr<AbstractCone<T>>& cone, Func&& func) {
    if (auto powerCone = std::dynamic_pointer_cast<PowerCone<T>>(cone)) {
        func(powerCone);
    } else if (auto expCone = std::dynamic_pointer_cast<ExponentialCone<T>>(cone)) {
        func(expCone);
    } else {
        std::cerr << "Unknown cone type" << std::endl;
    }
}

// Example usage
int main() {
    std::shared_ptr<AbstractCone<double>> cone = std::make_shared<PowerCone<double>>();
    conedispatch(cone, [](const auto& c) {
        std::cout << "Dispatched to cone type" << std::endl;
    });
    return 0;
}
