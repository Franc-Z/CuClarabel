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
#include <thrust/iterator/zip_iterator.h>

namespace Clarabel {

enum class PrimalOrDualCone {
    PrimalCone,
    DualCone
};

template <typename T>
class AbstractCone {
public:
    virtual ~AbstractCone() = default;
};

template <typename T>
class ZeroCone : public AbstractCone<T> {
public:
    explicit ZeroCone(int dim) : dim(dim) {
        if (dim < 1) {
            throw std::domain_error("dimension must be positive");
        }
    }

    int get_dim() const {
        return dim;
    }

private:
    int dim;
};

template <typename T>
class NonnegativeCone : public AbstractCone<T> {
public:
    explicit NonnegativeCone(int dim) : dim(dim), w(dim, 0), λ(dim, 0) {
        if (dim < 0) {
            throw std::domain_error("dimension must be nonnegative");
        }
    }

    int get_dim() const {
        return dim;
    }

    const std::vector<T>& get_w() const {
        return w;
    }

    const std::vector<T>& get_λ() const {
        return λ;
    }

private:
    int dim;
    std::vector<T> w;
    std::vector<T> λ;
};

template <typename T>
class SecondOrderConeSparseData {
public:
    explicit SecondOrderConeSparseData(int dim) : u(dim, 0), v(dim, 0), d(0) {}

    const std::vector<T>& get_u() const {
        return u;
    }

    const std::vector<T>& get_v() const {
        return v;
    }

    T get_d() const {
        return d;
    }

private:
    std::vector<T> u;
    std::vector<T> v;
    T d;
};

template <typename T>
class SecondOrderCone : public AbstractCone<T> {
public:
    explicit SecondOrderCone(int dim) : dim(dim), w(dim, 0), λ(dim, 0), η(0) {
        if (dim < 2) {
            throw std::domain_error("dimension must be >= 2");
        }

        if (dim > SOC_NO_EXPANSION_MAX_SIZE) {
            sparse_data = std::make_unique<SecondOrderConeSparseData<T>>(dim);
        }
    }

    int get_dim() const {
        return dim;
    }

    const std::vector<T>& get_w() const {
        return w;
    }

    const std::vector<T>& get_λ() const {
        return λ;
    }

    T get_η() const {
        return η;
    }

    const SecondOrderConeSparseData<T>* get_sparse_data() const {
        return sparse_data.get();
    }

private:
    int dim;
    std::vector<T> w;
    std::vector<T> λ;
    T η;
    std::unique_ptr<SecondOrderConeSparseData<T>> sparse_data;
};

template <typename T>
class PSDConeData {
public:
    explicit PSDConeData(int n) : λ(n, 0), Λisqrt(n, 0), R(n, std::vector<T>(n, 0)), Rinv(n, std::vector<T>(n, 0)), Hs(triangular_number(n), std::vector<T>(triangular_number(n), 0)), workmat1(n, std::vector<T>(n, 0)), workmat2(n, std::vector<T>(n, 0)), workmat3(n, std::vector<T>(n, 0)), workvec(triangular_number(n), 0) {}

    const std::vector<T>& get_λ() const {
        return λ;
    }

    const std::vector<T>& get_Λisqrt() const {
        return Λisqrt;
    }

    const std::vector<std::vector<T>>& get_R() const {
        return R;
    }

    const std::vector<std::vector<T>>& get_Rinv() const {
        return Rinv;
    }

    const std::vector<std::vector<T>>& get_Hs() const {
        return Hs;
    }

    const std::vector<std::vector<T>>& get_workmat1() const {
        return workmat1;
    }

    const std::vector<std::vector<T>>& get_workmat2() const {
        return workmat2;
    }

    const std::vector<std::vector<T>>& get_workmat3() const {
        return workmat3;
    }

    const std::vector<T>& get_workvec() const {
        return workvec;
    }

private:
    std::vector<T> λ;
    std::vector<T> Λisqrt;
    std::vector<std::vector<T>> R;
    std::vector<std::vector<T>> Rinv;
    std::vector<std::vector<T>> Hs;
    std::vector<std::vector<T>> workmat1;
    std::vector<std::vector<T>> workmat2;
    std::vector<std::vector<T>> workmat3;
    std::vector<T> workvec;
};

template <typename T>
class PSDTriangleCone : public AbstractCone<T> {
public:
    explicit PSDTriangleCone(int n) : n(n), numel(triangular_number(n)), data(n) {
        if (n < 0) {
            throw std::domain_error("dimension must be non-negative");
        }
    }

    int get_n() const {
        return n;
    }

    int get_numel() const {
        return numel;
    }

    const PSDConeData<T>& get_data() const {
        return data;
    }

private:
    int n;
    int numel;
    PSDConeData<T> data;
};

template <typename T>
class ExponentialCone : public AbstractCone<T> {
public:
    ExponentialCone() : H_dual(3, std::vector<T>(3, 0)), Hs(3, std::vector<T>(3, 0)), grad(3, 0), z(3, 0) {}

    const std::vector<std::vector<T>>& get_H_dual() const {
        return H_dual;
    }

    const std::vector<std::vector<T>>& get_Hs() const {
        return Hs;
    }

    const std::vector<T>& get_grad() const {
        return grad;
    }

    const std::vector<T>& get_z() const {
        return z;
    }

private:
    std::vector<std::vector<T>> H_dual;
    std::vector<std::vector<T>> Hs;
    std::vector<T> grad;
    std::vector<T> z;
};

template <typename T>
class PowerCone : public AbstractCone<T> {
public:
    explicit PowerCone(T α) : α(α), H_dual(3, std::vector<T>(3, 0)), Hs(3, std::vector<T>(3, 0)), grad(3, 0), z(3, 0) {}

    T get_α() const {
        return α;
    }

    const std::vector<std::vector<T>>& get_H_dual() const {
        return H_dual;
    }

    const std::vector<std::vector<T>>& get_Hs() const {
        return Hs;
    }

    const std::vector<T>& get_grad() const {
        return grad;
    }

    const std::vector<T>& get_z() const {
        return z;
    }

private:
    T α;
    std::vector<std::vector<T>> H_dual;
    std::vector<std::vector<T>> Hs;
    std::vector<T> grad;
    std::vector<T> z;
};

template <typename T>
class GenPowerConeData {
public:
    GenPowerConeData(const std::vector<T>& α, int dim2) : grad(α.size() + dim2, 0), z(α.size() + dim2, 0), μ(1), p(α.size() + dim2, 0), q(α.size(), 0), r(dim2, 0), d1(α.size(), 0), d2(0), ψ(1 / std::inner_product(α.begin(), α.end(), α.begin(), 0.0)), work(α.size() + dim2, 0), work_pb(α.size() + dim2, 0) {}

    const std::vector<T>& get_grad() const {
        return grad;
    }

    const std::vector<T>& get_z() const {
        return z;
    }

    T get_μ() const {
        return μ;
    }

    const std::vector<T>& get_p() const {
        return p;
    }

    const std::vector<T>& get_q() const {
        return q;
    }

    const std::vector<T>& get_r() const {
        return r;
    }

    const std::vector<T>& get_d1() const {
        return d1;
    }

    T get_d2() const {
        return d2;
    }

    T get_ψ() const {
        return ψ;
    }

    const std::vector<T>& get_work() const {
        return work;
    }

    const std::vector<T>& get_work_pb() const {
        return work_pb;
    }

private:
    std::vector<T> grad;
    std::vector<T> z;
    T μ;
    std::vector<T> p;
    std::vector<T> q;
    std::vector<T> r;
    std::vector<T> d1;
    T d2;
    T ψ;
    std::vector<T> work;
    std::vector<T> work_pb;
};

template <typename T>
class GenPowerCone : public AbstractCone<T> {
public:
    GenPowerCone(const std::vector<T>& α, int dim2) : α(α), dim2(dim2), data(α, dim2) {}

    const std::vector<T>& get_α() const {
        return α;
    }

    int get_dim2() const {
        return dim2;
    }

    const GenPowerConeData<T>& get_data() const {
        return data;
    }

private:
    std::vector<T> α;
    int dim2;
    GenPowerConeData<T> data;
};

const std::map<std::type_index, std::type_index> ConeDict = {
    {typeid(ZeroConeT), typeid(ZeroCone)},
    {typeid(NonnegativeConeT), typeid(NonnegativeCone)},
    {typeid(SecondOrderConeT), typeid(SecondOrderCone)},
    {typeid(ExponentialConeT), typeid(ExponentialCone)},
    {typeid(PowerConeT), typeid(PowerCone)},
    {typeid(GenPowerConeT), typeid(GenPowerCone)},
    {typeid(PSDTriangleConeT), typeid(PSDTriangleCone)}
};

} // namespace Clarabel
