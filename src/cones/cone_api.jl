#include <vector>
#include <map>
#include <cassert>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
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

namespace Clarabel {

enum class SupportedCone {
    ZeroConeT,
    NonnegativeConeT,
    SecondOrderConeT,
    ExponentialConeT,
    PowerConeT,
    GenPowerConeT,
    PSDTriangleConeT
};

struct ZeroConeT {
    int dim;
};

struct NonnegativeConeT {
    int dim;
};

struct SecondOrderConeT {
    int dim;
};

struct PowerConeT {
    float alpha;
};

struct GenPowerConeT {
    std::vector<float> alpha;
    int dim2;

    GenPowerConeT(const std::vector<float>& alpha, int dim2) : alpha(alpha), dim2(dim2) {
        assert(std::all_of(alpha.begin(), alpha.end(), [](float a) { return a > 0; }));
        assert(std::abs(std::accumulate(alpha.begin(), alpha.end(), 0.0f) - 1.0f) < std::numeric_limits<float>::epsilon() * alpha.size() / 2);
    }
};

struct ExponentialConeT {};

struct PSDTriangleConeT {
    int dim;
};

int nvars(const SupportedCone& cone) {
    switch (cone) {
        case SupportedCone::PSDTriangleConeT:
            return triangular_number(cone.dim);
        case SupportedCone::ExponentialConeT:
        case SupportedCone::PowerConeT:
            return 3;
        case SupportedCone::GenPowerConeT:
            return cone.alpha.size() + cone.dim2;
        default:
            return cone.dim;
    }
}

template <typename T>
auto make_cone(const SupportedCone& coneT) {
    using namespace std::placeholders;
    static const std::map<SupportedCone, std::function<AbstractCone<T>(const SupportedCone&)>> ConeDict = {
        {SupportedCone::ZeroConeT, [](const SupportedCone& coneT) { return ZeroCone<T>(coneT.dim); }},
        {SupportedCone::NonnegativeConeT, [](const SupportedCone& coneT) { return NonnegativeCone<T>(coneT.dim); }},
        {SupportedCone::SecondOrderConeT, [](const SupportedCone& coneT) { return SecondOrderCone<T>(coneT.dim); }},
        {SupportedCone::ExponentialConeT, [](const SupportedCone&) { return ExponentialCone<T>(); }},
        {SupportedCone::PowerConeT, [](const SupportedCone& coneT) { return PowerCone<T>(coneT.alpha); }},
        {SupportedCone::GenPowerConeT, [](const SupportedCone& coneT) { return GenPowerCone<T>(coneT.alpha, coneT.dim2); }},
        {SupportedCone::PSDTriangleConeT, [](const SupportedCone& coneT) { return PSDTriangleCone<T>(coneT.dim); }}
    };

    return ConeDict.at(coneT)(coneT);
}

} // namespace Clarabel
