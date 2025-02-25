#include <vector>
#include <cassert>
#include <Eigen/Sparse>
#include <Eigen/Dense>

using namespace Eigen;

int count_soc(const SupportedCone& cone, int size_soc) {
    int numel_cone = Clarabel::nvars(cone);
    assert(numel_cone > size_soc);

    int num_socs = 1;
    numel_cone -= size_soc - 1;

    while (numel_cone > size_soc - 1) {
        numel_cone -= size_soc - 2;
        num_socs += 1;
    }

    num_socs += 1;

    return num_socs, numel_cone + 1;
}

template <typename T>
void augment_data(const SparseMatrix<T>& At0, std::vector<T>& b0, const std::vector<int>& rng_row, int size_soc, int num_soc, int last_size, int& augx_idx, SparseMatrix<T>& Atnew, std::vector<T>& bnew, std::vector<SupportedCone>& conenew) {
    SparseMatrix<T> At = At0.middleCols(rng_row[0], rng_row.size());
    std::vector<T> b(rng_row.size());
    for (size_t i = 0; i < rng_row.size(); ++i) {
        b[i] = b0[rng_row[i]];
    }

    int n = At.rows();
    int m = At.cols();
    int reduce_soc = size_soc - 2;
    assert(reduce_soc > 0);

    bnew.reserve(m + 2 * (num_soc - 1));
    conenew.reserve(num_soc);

    Atnew = At.middleCols(0, 1);
    bnew.push_back(b[0]);
    int idx = 1;

    for (int i = 1; i <= num_soc; ++i) {
        if (i == num_soc) {
            std::vector<int> rng(idx + 1, idx + last_size - 1);
            Atnew.conservativeResize(Atnew.rows(), Atnew.cols() + rng.size());
            Atnew.middleCols(Atnew.cols() - rng.size(), rng.size()) = At.middleCols(rng[0], rng.size());
            bnew.insert(bnew.end(), b.begin() + rng[0], b.begin() + rng[0] + rng.size());
            conenew.push_back(Clarabel::SecondOrderConeT(last_size));
        } else {
            std::vector<int> rng(idx + 1, idx + reduce_soc);
            Atnew.conservativeResize(Atnew.rows(), Atnew.cols() + rng.size());
            Atnew.middleCols(Atnew.cols() - rng.size(), rng.size()) = At.middleCols(rng[0], rng.size());
            bnew.insert(bnew.end(), b.begin() + rng[0], b.begin() + rng[0] + rng.size());
            conenew.push_back(Clarabel::SecondOrderConeT(size_soc));

            idx += reduce_soc;
            augx_idx += 1;
            Atnew.conservativeResize(Atnew.rows(), Atnew.cols() + 2);
            Atnew.middleCols(Atnew.cols() - 2, 2) = SparseMatrix<T>(n, 2);
            Atnew.coeffRef(augx_idx, Atnew.cols() - 2) = -1;
            Atnew.coeffRef(augx_idx, Atnew.cols() - 1) = -1;
            bnew.push_back(0);
            bnew.push_back(0);
        }
    }
}

template <typename T>
void augment_A_b_soc(std::vector<SupportedCone>& cones, SparseMatrix<T>& P, std::vector<T>& q, SparseMatrix<T>& A, std::vector<T>& b, int size_soc, std::vector<int>& num_socs, std::vector<int>& last_sizes, std::vector<int>& soc_indices, std::vector<int>& soc_starts) {
    int m = A.rows();
    int n = A.cols();

    int extra_dim = std::accumulate(num_socs.begin(), num_socs.end(), 0) - num_socs.size();

    SparseMatrix<T> At(n + extra_dim, m);
    At.topRows(n) = A.transpose();
    At.bottomRows(extra_dim).setZero();

    std::vector<T> bnew;
    bnew.reserve(m + 2 * extra_dim);
    std::vector<SupportedCone> conesnew;
    conesnew.reserve(cones.size() + extra_dim);

    SparseMatrix<T> Atnew(n + extra_dim, 0);

    int start_idx = 0;
    int end_idx = 0;
    int cone_idx = 0;
    int augx_idx = n;

    for (size_t i = 0; i < soc_indices.size(); ++i) {
        int ind = soc_indices[i];

        conesnew.insert(conesnew.end(), cones.begin() + cone_idx, cones.begin() + ind);

        int numel_cone = Clarabel::nvars(cones[ind]);

        end_idx = soc_starts[i];

        std::vector<int> rng(start_idx + 1, end_idx);
        Atnew.conservativeResize(Atnew.rows(), Atnew.cols() + rng.size());
        Atnew.middleCols(Atnew.cols() - rng.size(), rng.size()) = At.middleCols(rng[0], rng.size());
        bnew.insert(bnew.end(), b.begin() + rng[0], b.begin() + rng[0] + rng.size());

        start_idx = end_idx;
        end_idx += numel_cone;
        std::vector<int> rng_cone(start_idx + 1, end_idx);

        SparseMatrix<T> Ati;
        std::vector<T> bi;
        std::vector<SupportedCone> conesi;
        augment_data(At, b, rng_cone, size_soc, num_socs[i], last_sizes[i], augx_idx, Ati, bi, conesi);

        Atnew.conservativeResize(Atnew.rows(), Atnew.cols() + Ati.cols());
        Atnew.middleCols(Atnew.cols() - Ati.cols(), Ati.cols()) = Ati;
        bnew.insert(bnew.end(), bi.begin(), bi.end());
        conesnew.insert(conesnew.end(), conesi.begin(), conesi.end());

        start_idx = end_idx;
        cone_idx = ind;
    }

    if (cone_idx < cones.size()) {
        std::vector<int> rng(start_idx + 1, A.cols());
        Atnew.conservativeResize(Atnew.rows(), Atnew.cols() + rng.size());
        Atnew.middleCols(Atnew.cols() - rng.size(), rng.size()) = At.middleCols(rng[0], rng.size());
        bnew.insert(bnew.end(), b.begin() + rng[0], b.end());
        conesnew.insert(conesnew.end(), cones.begin() + cone_idx, cones.end());
    }

    SparseMatrix<T> Pnew(n + extra_dim, n + extra_dim);
    Pnew.topLeftCorner(n, n) = P;
    Pnew.bottomRightCorner(extra_dim, extra_dim).setZero();
    Pnew.topRightCorner(n, extra_dim).setZero();
    Pnew.bottomLeftCorner(extra_dim, n).setZero();

    P = Pnew;
    q.resize(n + extra_dim, 0);
    A = Atnew.transpose();
    b = bnew;
    cones = conesnew;
}

void expand_soc(std::vector<SupportedCone>& cones, int size_soc, std::vector<int>& num_socs, std::vector<int>& last_sizes, std::vector<int>& soc_indices, std::vector<int>& soc_starts) {
    int n_large_soc = 0;
    soc_indices.reserve(cones.size());
    soc_starts.reserve(cones.size());
    num_socs.reserve(cones.size());
    last_sizes.reserve(cones.size());

    int cones_dim = 0;
    for (size_t i = 0; i < cones.size(); ++i) {
        int numel_cone = Clarabel::nvars(cones[i]);
        if (typeid(cones[i]) == typeid(Clarabel::SecondOrderConeT) && numel_cone > size_soc) {
            soc_indices.push_back(i);
            soc_starts.push_back(cones_dim);

            int num_soc, last_size;
            std::tie(num_soc, last_size) = count_soc(cones[i], size_soc);
            num_socs.push_back(num_soc);
            last_sizes.push_back(last_size);
            n_large_soc += 1;
        }

        cones_dim += numel_cone;
    }

    num_socs.resize(n_large_soc);
    last_sizes.resize(n_large_soc);
}
