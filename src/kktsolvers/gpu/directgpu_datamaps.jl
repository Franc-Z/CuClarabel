#include <vector>
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

class SparseExpansionFullMap {
public:
    virtual int pdim() const = 0;
    virtual int nnz_vec() const = 0;
};

int pdim(const std::vector<SparseExpansionFullMap*>& maps) {
    int sum = 0;
    for (const auto& map : maps) {
        sum += map->pdim();
    }
    return sum;
}

int nnz_vec(const std::vector<SparseExpansionFullMap*>& maps) {
    int sum = 0;
    for (const auto& map : maps) {
        sum += map->nnz_vec();
    }
    return sum;
}

class SOCExpansionFullMap : public SparseExpansionFullMap {
public:
    std::vector<int> u;        // off diag dense columns u
    std::vector<int> v;        // off diag dense columns v
    std::vector<int> ut;       // off diag dense columns ut
    std::vector<int> vt;       // off diag dense columns vt
    std::array<int, 2> D;      // diag D

    SOCExpansionFullMap(const SecondOrderCone& cone) {
        u.resize(cone.numel());
        v.resize(cone.numel());
        ut.resize(cone.numel());
        vt.resize(cone.numel());
        D = {0, 0};
    }

    int pdim() const override {
        return 2;
    }

    int nnz_vec() const override {
        return 4 * u.size();
    }

    std::array<int, 2> Dsigns() const {
        return {-1, 1};
    }
};

SOCExpansionFullMap expansion_fullmap(const SecondOrderCone& cone) {
    return SOCExpansionFullMap(cone);
}

void _csc_colcount_sparsecone_full(
    const SecondOrderCone& cone,
    SOCExpansionFullMap& map,
    SparseMatrixCSC& K,
    int row,
    int col
) {
    int nvars = cone.numel();

    _csc_colcount_colvec(K, nvars, row, col);     // v column
    _csc_colcount_colvec(K, nvars, row, col + 1); // u column
    _csc_colcount_rowvec(K, nvars, col, row);     // v row
    _csc_colcount_rowvec(K, nvars, col + 1, row); // u row

    _csc_colcount_diag(K, col, map.pdim());
}

void _csc_fill_sparsecone_full(
    const SecondOrderCone& cone,
    SOCExpansionFullMap& map,
    SparseMatrixCSC& K,
    int row,
    int col
) {
    _csc_fill_colvec(K, map.v, row, col);         // v
    _csc_fill_colvec(K, map.u, row, col + 1);     // u
    _csc_fill_rowvec(K, map.vt, col, row);        // vt
    _csc_fill_rowvec(K, map.ut, col + 1, row);    // ut

    _csc_fill_diag(K, map.D, col, map.pdim());
}

template <typename T>
void _csc_update_sparsecone_full(
    const SecondOrderCone<T>& cone,
    SOCExpansionFullMap& map,
    std::function<void(std::vector<int>&, const std::vector<int>&)> updateFcn,
    std::function<void(std::vector<int>&, T)> scaleFcn
) {
    T η2 = cone.η * cone.η;

    updateFcn(map.u, cone.sparse_data.u);
    updateFcn(map.v, cone.sparse_data.v);
    updateFcn(map.ut, cone.sparse_data.u);
    updateFcn(map.vt, cone.sparse_data.v);
    scaleFcn(map.u, -η2);
    scaleFcn(map.v, -η2);
    scaleFcn(map.ut, -η2);
    scaleFcn(map.vt, -η2);

    updateFcn(map.D, {-η2, +η2});
}

class GenPowExpansionFullMap : public SparseExpansionFullMap {
public:
    std::vector<int> p;        // off diag dense columns p
    std::vector<int> q;        // off diag dense columns q
    std::vector<int> r;        // off diag dense columns r
    std::vector<int> pt;       // off diag dense rows pt
    std::vector<int> qt;       // off diag dense rows qt
    std::vector<int> rt;       // off diag dense rows rt
    std::array<int, 3> D;      // diag D

    GenPowExpansionFullMap(const GenPowerCone& cone) {
        p.resize(cone.numel());
        q.resize(cone.dim1());
        r.resize(cone.dim2());
        pt.resize(cone.numel());
        qt.resize(cone.dim1());
        rt.resize(cone.dim2());
        D = {0, 0, 0};
    }

    int pdim() const override {
        return 3;
    }

    int nnz_vec() const override {
        return (p.size() + q.size() + r.size()) * 2;
    }

    std::array<int, 3> Dsigns() const {
        return {-1, -1, +1};
    }
};

GenPowExpansionFullMap expansion_fullmap(const GenPowerCone& cone) {
    return GenPowExpansionFullMap(cone);
}

void _csc_colcount_sparsecone_full(
    const GenPowerCone& cone,
    GenPowExpansionFullMap& map,
    SparseMatrixCSC& K,
    int row,
    int col
) {
    int nvars = cone.numel();
    int dim1 = cone.dim1();
    int dim2 = cone.dim2();

    _csc_colcount_colvec(K, dim1, row, col);         // q column
    _csc_colcount_colvec(K, dim2, row + dim1, col + 1); // r column
    _csc_colcount_colvec(K, nvars, row, col + 2);    // p column

    _csc_colcount_rowvec(K, dim1, col, row);         // qt row
    _csc_colcount_rowvec(K, dim2, col + 1, row + dim1); // rt row
    _csc_colcount_rowvec(K, nvars, col + 2, row);    // pt row

    _csc_colcount_diag(K, col, map.pdim());
}

template <typename T>
void _csc_fill_sparsecone_full(
    const GenPowerCone<T>& cone,
    GenPowExpansionFullMap& map,
    SparseMatrixCSC<T>& K,
    int row,
    int col
) {
    int dim1 = cone.dim1();

    _csc_fill_colvec(K, map.q, row, col);         // q
    _csc_fill_colvec(K, map.r, row + dim1, col + 1); // r
    _csc_fill_colvec(K, map.p, row, col + 2);     // p

    _csc_fill_rowvec(K, map.qt, col, row);        // qt
    _csc_fill_rowvec(K, map.rt, col + 1, row + dim1); // rt
    _csc_fill_rowvec(K, map.pt, col + 2, row);    // pt

    _csc_fill_diag(K, map.D, col, map.pdim());
}

template <typename T>
void _csc_update_sparsecone_full(
    const GenPowerCone<T>& cone,
    GenPowExpansionFullMap& map,
    std::function<void(std::vector<int>&, const std::vector<int>&)> updateFcn,
    std::function<void(std::vector<int>&, T)> scaleFcn
) {
    const auto& data = cone.data;
    T sqrtμ = std::sqrt(data.μ);

    updateFcn(map.q, data.q);
    updateFcn(map.r, data.r);
    updateFcn(map.p, data.p);
    updateFcn(map.qt, data.q);
    updateFcn(map.rt, data.r);
    updateFcn(map.pt, data.p);
    scaleFcn(map.q, -sqrtμ);
    scaleFcn(map.r, -sqrtμ);
    scaleFcn(map.p, -sqrtμ);
    scaleFcn(map.qt, -sqrtμ);
    scaleFcn(map.rt, -sqrtμ);
    scaleFcn(map.pt, -sqrtμ);

    updateFcn(map.D, {-1, -1, 1});
}

class FullDataMap {
public:
    std::vector<int> P;
    std::vector<int> A;
    std::vector<int> At;        // YC: not sure whether we need it or not
    std::vector<int> Hsblocks;  // indices of the lower RHS blocks (by cone)
    std::vector<SparseExpansionFullMap*> sparse_maps; // sparse cone expansion terms

    std::vector<int> diagP;
    std::vector<int> diag_full;

    FullDataMap(const SparseMatrixCSC& Pmat, const SparseMatrixCSC& Amat, const std::vector<SupportedCone>& cones) {
        int m = Amat.rows();
        int n = Pmat.cols();
        P.resize(Pmat.nonZeros());
        A.resize(Amat.nonZeros());
        At.resize(Amat.nonZeros());

        diagP.resize(n);

        Hsblocks = _allocate_kkt_Hsblocks<int>(cones);

        int nsparse = std::count_if(cones.begin(), cones.end(), [](const SupportedCone& cone) {
            return is_sparse_expandable(cone);
        });
        sparse_maps.reserve(nsparse);

        for (const auto& cone : cones) {
            if (is_sparse_expandable(cone)) {
                sparse_maps.push_back(expansion_fullmap(cone));
            }
        }

        diag_full.resize(m + n + pdim(sparse_maps));
    }
};

class GPUDataMap {
public:
    thrust::device_vector<int> P;
    thrust::device_vector<int> A;
    thrust::device_vector<int> At;        // YC: not sure whether we need it or not
    thrust::device_vector<int> Hsblocks;  // indices of the lower RHS blocks (by cone)
    // thrust::device_vector<SparseExpansionFullMap*> sparse_maps; // YC: disabled sparse cone expansion terms

    thrust::device_vector<int> diagP;
    thrust::device_vector<int> diag_full;

    GPUDataMap(const SparseMatrixCSC& Pmat, const SparseMatrixCSC& Amat, const std::vector<SupportedCone>& cones, const FullDataMap& mapcpu) {
        int m = Amat.rows();
        int n = Pmat.cols();
        P = mapcpu.P.empty() ? thrust::device_vector<int>(0) : thrust::device_vector<int>(mapcpu.P.begin(), mapcpu.P.end());
        A = mapcpu.A.empty() ? thrust::device_vector<int>(0) : thrust::device_vector<int>(mapcpu.A.begin(), mapcpu.A.end());
        At = mapcpu.At.empty() ? thrust::device_vector<int>(0) : thrust::device_vector<int>(mapcpu.At.begin(), mapcpu.At.end());

        diagP = thrust::device_vector<int>(mapcpu.diagP.begin(), mapcpu.diagP.end());

        Hsblocks = thrust::device_vector<int>(mapcpu.Hsblocks.begin(), mapcpu.Hsblocks.end());

        diag_full = thrust::device_vector<int>(mapcpu.diag_full.begin(), mapcpu.diag_full.end());
    }
};
