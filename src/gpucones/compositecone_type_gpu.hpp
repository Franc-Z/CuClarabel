#include <vector>
#include <unordered_map>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <typeindex>

template <typename T>
class CompositeConeGPU {
public:
    // Redundant CPU data, need to be removed later
    std::vector<AbstractCone<T>*> cones;

    // Count of each cone type
    std::unordered_map<std::type_index, int> type_counts;

    // Overall size of the composite cone
    int numel;
    int degree;

    // Range views
    std::vector<std::pair<int, int>> rng_cones;
    std::vector<std::pair<int, int>> rng_blocks;

    // The flag for symmetric cone check
    bool _is_symmetric;
    int n_linear;
    int n_nn;
    int n_soc;
    int n_exp;
    int n_pow;
    int n_psd;

    std::vector<int> idx_eq;
    std::vector<int> idx_inq;

    // Data
    thrust::device_vector<T> w;
    thrust::device_vector<T> λ;
    thrust::device_vector<T> η;

    // Nonsymmetric cone
    thrust::device_vector<T> αp;           // Power parameters of power cones
    thrust::device_vector<T> H_dual;        // Hessian of the dual barrier at z 
    thrust::device_vector<T> Hs;            // Scaling matrix
    thrust::device_vector<T> grad;         // Gradient of the dual barrier at z 

    // PSD cone
    int psd_dim;                  // We only support PSD cones with the same small dimension
    thrust::device_vector<T> chol1;
    thrust::device_vector<T> chol2;
    thrust::device_vector<T> SVD;
    thrust::device_vector<T> λpsd;
    thrust::device_vector<T> Λisqrt;
    thrust::device_vector<T> R;
    thrust::device_vector<T> Rinv;
    thrust::device_vector<T> Hspsd;

    // Workspace for various internal uses
    thrust::device_vector<T> workmat1;
    thrust::device_vector<T> workmat2;
    thrust::device_vector<T> workmat3;
    thrust::device_vector<T> workvec;

    // Step size
    thrust::device_vector<T> α;

    CompositeConeGPU(CompositeCone<T>& cpucones) {
        // Information from the CompositeCone on CPU 
        cones = cpucones.cones;
        type_counts = cpucones.type_counts;
        _is_symmetric = cpucones._is_symmetric;

        n_zero = type_counts.count(typeid(ZeroCone)) ? type_counts[typeid(ZeroCone)] : 0;
        n_nn = type_counts.count(typeid(NonnegativeCone)) ? type_counts[typeid(NonnegativeCone)] : 0;
        n_linear = n_zero + n_nn;
        n_soc = type_counts.count(typeid(SecondOrderCone)) ? type_counts[typeid(SecondOrderCone)] : 0;
        n_exp = type_counts.count(typeid(ExponentialCone)) ? type_counts[typeid(ExponentialCone)] : 0;
        n_pow = type_counts.count(typeid(PowerCone)) ? type_counts[typeid(PowerCone)] : 0;
        n_psd = type_counts.count(typeid(PSDTriangleCone)) ? type_counts[typeid(PSDTriangleCone)] : 0;

        // idx set for eq and ineq constraints
        for (int i = 0; i < n_linear; ++i) {
            if (typeid(cones[i]) == typeid(ZeroCone<T>)) {
                idx_eq.push_back(i);
            } else {
                idx_inq.push_back(i);
            }
        }

        // Count up elements and degree
        numel = std::accumulate(cones.begin(), cones.end(), 0, [](int sum, AbstractCone<T>* cone) {
            return sum + cone->numel();
        });
        degree = std::accumulate(cones.begin(), cones.end(), 0, [](int sum, AbstractCone<T>* cone) {
            return sum + cone->degree();
        });

        int numel_linear = std::accumulate(cones.begin(), cones.begin() + n_linear, 0, [](int sum, AbstractCone<T>* cone) {
            return sum + cone->numel();
        });
        int max_linear = std::max_element(cones.begin(), cones.begin() + n_linear, [](AbstractCone<T>* a, AbstractCone<T>* b) {
            return a->numel() < b->numel();
        })->numel();
        int numel_soc = std::accumulate(cones.begin(), cones.begin() + n_linear + n_soc, 0, [](int sum, AbstractCone<T>* cone) {
            return sum + cone->numel();
        });

        w.resize(numel_linear + numel_soc);
        λ.resize(numel_linear + numel_soc);
        η.resize(n_soc);

        // Initialize space for nonsymmetric cones
        αp.resize(n_pow);
        int pow_ind = n_linear + n_soc + n_exp;
        // Store the power parameter of each power cone
        for (int i = 0; i < n_pow; ++i) {
            αp[i] = cones[i + pow_ind]->α;
        }

        H_dual.resize((n_exp + n_pow) * 3 * 3);
        Hs.resize((n_exp + n_pow) * 3 * 3);
        grad.resize((n_exp + n_pow) * 3);

        // PSD cone
        // We require all psd cones have the same dimensionality
        int psd_ind = pow_ind + n_pow;
        psd_dim = type_counts.count(typeid(PSDTriangleCone)) ? cones[psd_ind]->n : 0;
        for (int i = 0; i < n_psd; ++i) {
            if (psd_dim != cones[psd_ind + i]->n) {
                throw std::runtime_error("Not all positive definite cones have the same dimensionality!");
            }
        }

        chol1.resize(psd_dim * psd_dim * n_psd);
        chol2.resize(psd_dim * psd_dim * n_psd);
        SVD.resize(psd_dim * psd_dim * n_psd);

        λpsd.resize(psd_dim * n_psd);
        Λisqrt.resize(psd_dim * n_psd);
        R.resize(psd_dim * psd_dim * n_psd);
        Rinv.resize(psd_dim * psd_dim * n_psd);
        Hspsd.resize(triangular_number(psd_dim) * triangular_number(psd_dim) * n_psd);

        workmat1.resize(psd_dim * psd_dim * n_psd);
        workmat2.resize(psd_dim * psd_dim * n_psd);
        workmat3.resize(psd_dim * psd_dim * n_psd);
        workvec.resize(triangular_number(psd_dim) * n_psd);

        α.resize(std::accumulate(cones.begin(), cones.end(), 0, [](int sum, AbstractCone<T>* cone) {
            return sum + cone->numel();
        })); // Workspace for step size calculation and neighborhood check

        rng_cones = cpucones.rng_cones;
        rng_blocks = cpucones.rng_blocks;
    }

    AbstractCone<T>* operator[](int i) const {
        return cones[i];
    }

    std::vector<AbstractCone<T>*> operator[](const std::vector<bool>& b) const {
        std::vector<AbstractCone<T>*> result;
        for (size_t i = 0; i < cones.size(); ++i) {
            if (b[i]) {
                result.push_back(cones[i]);
            }
        }
        return result;
    }

    auto begin() const {
        return cones.begin();
    }

    auto end() const {
        return cones.end();
    }

    size_t size() const {
        return cones.size();
    }

    auto eachindex() const {
        return cones.eachindex();
    }

    static int get_type_count(const CompositeConeGPU<T>& cones, const std::type_index& type) {
        if (cones.type_counts.count(type)) {
            return cones.type_counts.at(type);
        } else {
            return 0;
        }
    }
};
