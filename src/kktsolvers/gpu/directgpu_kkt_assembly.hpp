#include <vector>
#include <thrust/device_vector.h>
#include <cusparse.h>
#include <cuda_runtime.h>
#include "SparseMatrixCSC.hpp"
#include "CompositeCone.hpp"
#include "FullDataMap.hpp"

template <typename T>
std::pair<SparseMatrixCSC<T>, FullDataMap> _assemble_full_kkt_matrix(
    const SparseMatrixCSC<T>& P,
    const SparseMatrixCSC<T>& A,
    const CompositeCone<T>& cones,
    const std::string& shape = "triu"  // or "tril"
) {
    FullDataMap map(P, A, cones);
    int m = A.rows();
    int n = P.cols();
    int p = pdim(map.sparse_maps);

    // entries actually on the diagonal of P
    int nnz_diagP = _count_diagonal_entries_full(P);

    // total entries in the Hs blocks
    int nnz_Hsblocks = map.Hsblocks.size();

    int nnzKKT = (P.nonZeros() +      // Number of elements in P
    n -                     // Number of elements in diagonal top left block
    nnz_diagP +             // remove double count on the diagonal if P has entries
    2 * A.nonZeros() +                // Number of nonzeros in A and A'
    nnz_Hsblocks +          // Number of elements in diagonal below A'
    2 * nnz_vec(map.sparse_maps) + // Number of elements in sparse cone off diagonals, 2x compared to the triangle form
    p                       // Number of elements in diagonal of sparse cones
    );

    SparseMatrixCSC<T> K = _csc_spalloc<T>(m + n + p, m + n + p, nnzKKT);

    _full_kkt_assemble_colcounts(K, P, A, cones, map);
    _full_kkt_assemble_fill(K, P, A, cones, map);

    return std::make_pair(K, map);
}

template <typename T>
void _full_kkt_assemble_colcounts(
    SparseMatrixCSC<T>& K,
    const SparseMatrixCSC<T>& P,
    const SparseMatrixCSC<T>& A,
    const CompositeCone<T>& cones,
    FullDataMap& map
) {
    int m = A.rows();
    int n = A.cols();

    // use K.colptr to hold nnz entries in each
    // column of the KKT matrix
    std::fill(K.colptr.begin(), K.colptr.end(), 0);

    // Count first n columns of KKT
    _csc_colcount_block_full(K, P, A, 1);
    _csc_colcount_missing_diag_full(K, P, 1);
    _csc_colcount_block(K, A, n + 1, 'T');

    // track the next sparse column to fill (assuming triu fill)
    int pcol = m + n + 1; // next sparse column to fill
    auto sparse_map_iter = map.sparse_maps.begin();

    for (size_t i = 0; i < cones.size(); ++i) {
        int row = cones.rng_cones[i].first + n;

        // add the Hs blocks in the lower right
        int blockdim = cones[i].numel();
        if (Hs_is_diagonal(cones[i])) {
            _csc_colcount_diag(K, row, blockdim);
        } else {
            _csc_colcount_dense_full(K, row, blockdim);
        }

        // add sparse expansions columns for sparse cones 
        if (is_sparse_expandable(cones[i])) {
            auto thismap = *sparse_map_iter++;
            _csc_colcount_sparsecone_full(cones[i], thismap, K, row, pcol);
            pcol += pdim(thismap); // next sparse column to fill 
        }
    }
}

template <typename T>
void _full_kkt_assemble_fill(
    SparseMatrixCSC<T>& K,
    const SparseMatrixCSC<T>& P,
    const SparseMatrixCSC<T>& A,
    const CompositeCone<T>& cones,
    FullDataMap& map
) {
    int m = A.rows();
    int n = A.cols();

    // cumsum total entries to convert to K.colptr
    _csc_colcount_to_colptr(K);

    // filling [P At;A 0] parts
    _csc_fill_P_block_with_missing_diag_full(K, P, map.P);
    _csc_fill_block(K, A, map.A, n + 1, 1, 'N');
    _csc_fill_block(K, A, map.At, 1, n + 1, 'T');

    // track the next sparse column to fill (assuming full fill)
    int pcol = m + n + 1; // next sparse column to fill
    auto sparse_map_iter = map.sparse_maps.begin();

    for (size_t i = 0; i < cones.size(); ++i) {
        int row = cones.rng_cones[i].first + n;

        // add the Hs blocks in the lower right
        int blockdim = cones[i].numel();
        auto block = std::span(map.Hsblocks.begin() + cones.rng_blocks[i].first, map.Hsblocks.begin() + cones.rng_blocks[i].second);

        if (Hs_is_diagonal(cones[i])) {
            _csc_fill_diag(K, block, row, blockdim);
        } else {
            _csc_fill_dense_full(K, block, row, blockdim);
        }

        // add sparse expansions columns for sparse cones 
        if (is_sparse_expandable(cones[i])) {
            auto thismap = *sparse_map_iter++;
            _csc_fill_sparsecone_full(cones[i], thismap, K, row, pcol);
            pcol += pdim(thismap); // next sparse column to fill 
        }
    }

    // backshift the colptrs to recover K.colptr again
    _kkt_backshift_colptrs(K);

    // Now we can populate the index of the full diagonal.
    // We have filled in structural zeros on it everywhere.

    _map_diag_full(K, map.diag_full);
    std::copy(map.diag_full.begin(), map.diag_full.begin() + n, map.diagP.begin());
}
