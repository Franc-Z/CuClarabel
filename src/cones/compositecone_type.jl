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

template <typename T>
class CompositeCone {
public:
    CompositeCone(const std::vector<SupportedCone>& cone_specs, bool use_gpu) {
        cones.reserve(cone_specs.size());

        // Assumed symmetric to start
        _is_symmetric = true;

        // Create cones with the given dims
        for (const auto& coneT : cone_specs) {
            // Make a new cone
            auto cone = make_cone<T>(coneT);

            // Update global problem symmetry
            _is_symmetric = _is_symmetric && is_symmetric(cone);

            // Increment type counts
            auto key = ConeDict.at(typeid(coneT));
            type_counts[key] += 1;

            cones.push_back(cone);
        }

        // Count up elements and degree
        numel = std::accumulate(cones.begin(), cones.end(), 0, [](int sum, const auto& cone) {
            return sum + Clarabel::numel(cone);
        });
        degree = std::accumulate(cones.begin(), cones.end(), 0, [](int sum, const auto& cone) {
            return sum + Clarabel::degree(cone);
        });

        // Ranges for the subvectors associated with each cone,
        // and the range for the corresponding entries
        // in the Hs sparse block
        rng_cones = collect(rng_cones_iterator(cones));
        rng_blocks = collect(rng_blocks_iterator(cones, use_gpu));
    }

    const auto& get_cones() const {
        return cones;
    }

    int get_numel() const {
        return numel;
    }

    int get_degree() const {
        return degree;
    }

    const auto& get_rng_cones() const {
        return rng_cones;
    }

    const auto& get_rng_blocks() const {
        return rng_blocks;
    }

    bool is_symmetric() const {
        return _is_symmetric;
    }

    int get_type_count(const std::type_info& type) const {
        auto it = type_counts.find(type);
        if (it != type_counts.end()) {
            return it->second;
        }
        return 0;
    }

private:
    std::vector<AbstractCone<T>> cones;
    std::map<std::type_index, int> type_counts;
    int numel;
    int degree;
    std::vector<std::pair<int, int>> rng_cones;
    std::vector<std::pair<int, int>> rng_blocks;
    bool _is_symmetric;
};

template <typename T>
class RangeConesIterator {
public:
    RangeConesIterator(const std::vector<AbstractCone<T>>& cones) : cones(cones) {}

    auto begin() const {
        return cones.begin();
    }

    auto end() const {
        return cones.end();
    }

private:
    const std::vector<AbstractCone<T>>& cones;
};

template <typename T>
class RangeBlocksIterator {
public:
    RangeBlocksIterator(const std::vector<AbstractCone<T>>& cones) : cones(cones) {}

    auto begin() const {
        return cones.begin();
    }

    auto end() const {
        return cones.end();
    }

private:
    const std::vector<AbstractCone<T>>& cones;
};

template <typename T>
RangeConesIterator<T> rng_cones_iterator(const std::vector<AbstractCone<T>>& cones) {
    return RangeConesIterator<T>(cones);
}

template <typename T>
RangeBlocksIterator<T> rng_blocks_iterator(const std::vector<AbstractCone<T>>& cones, bool use_gpu) {
    return use_gpu ? RangeBlocksIteratorFull<T>(cones) : RangeBlocksIterator<T>(cones);
}

template <typename T>
class RangeBlocksIteratorFull {
public:
    RangeBlocksIteratorFull(const std::vector<AbstractCone<T>>& cones) : cones(cones) {}

    auto begin() const {
        return cones.begin();
    }

    auto end() const {
        return cones.end();
    }

private:
    const std::vector<AbstractCone<T>>& cones;
};

template <typename T>
RangeBlocksIteratorFull<T> rng_blocks_iterator_full(const std::vector<AbstractCone<T>>& cones) {
    return RangeBlocksIteratorFull<T>(cones);
}

} // namespace Clarabel
