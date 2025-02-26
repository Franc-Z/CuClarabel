#ifndef GPU_DEFAULTS_HPP
#define GPU_DEFAULTS_HPP

#include <vector>
#include <thrust/device_vector.h>
#include <stdexcept>
#include <unordered_map>
#include <variant>

template <typename T>
class AbstractGPUSolver {
public:
    virtual ~AbstractGPUSolver() = default;

    virtual std::string required_matrix_shape() const {
        throw std::runtime_error("function not implemented");
    }

    virtual void update_values(
        const std::vector<int>& index,
        const std::vector<T>& values
    ) {
        throw std::runtime_error("function not implemented");
    }

    virtual void scale_values(
        const std::vector<int>& index,
        T scale
    ) {
        throw std::runtime_error("function not implemented");
    }

    virtual void refactor() {
        throw std::runtime_error("function not implemented");
    }

    virtual void solve(
        std::vector<T>& x,
        const std::vector<T>& b
    ) {
        throw std::runtime_error("function not implemented");
    }
};

using GPUSolversDict = std::unordered_map<std::string, std::variant<AbstractGPUSolver<float>, AbstractGPUSolver<double>>>;

#endif // GPU_DEFAULTS_HPP
