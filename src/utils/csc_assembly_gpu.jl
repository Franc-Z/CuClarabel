using SparseArrays

function _csr_spalloc_gpu(T::Type{<:AbstractFloat}, n::Cint, nnz::Cint)

    rowptr = CUDA.zeros(Cint,n+1)
    colval = CUDA.zeros(Cint,nnz)
    nzval  = CUDA.zeros(T,nnz)

    return rowptr, colval, nzval
end

# #increment the K.colptr by the number of nonzeros
# #in a dense upper/lower triangle on the diagonal.
# function _csr_colcount_dense_triangle(K,initcol,blockcols,shape)
#     cols  = initcol:(initcol + (blockcols - 1))
#     if shape === :triu
#         @views K.colptr[cols] += 1:blockcols
#     else
#         @views K.colptr[cols] += blockcols:-1:1
#     end
# end

#increment the K.colptr by the number of nonzeros
#in a dense block on the diagonal.
function _kernel_csr_rowcount_dense_full(
    rowptr::AbstractVector{Cint}, 
    rng_cones::AbstractVector, 
    row_shift::Cint, 
    n_shift::Cint,
    n_cones::Cint
)

    i = (blockIdx().x-one(Cint))*blockDim().x+threadIdx().x

    #just add the nonzero count each row
    if i <= n_cones
        rng_cones_i = rng_cones[n_shift+i] .+ (row_shift + 1)
        len_i = length(rng_cones_i)

        @inbounds for idx in rng_cones_i
            rowptr[idx] += len_i            #add blockdim for each row
        end
    end

    return nothing
end

@inline function _csr_rowcount_dense_full_gpu(
    rowptr::AbstractVector{Cint}, 
    rng_cones::AbstractVector, 
    row_shift::Cint, 
    n_shift::Cint,
    n_cones::Cint
)
    @assert(length(rng_cones) == n_shift+n_cones)

    kernel = @cuda launch=false _kernel_csr_rowcount_dense_full(rowptr, rng_cones, row_shift, n_shift, n_cones)
    config = launch_configuration(kernel.fun)
    threads = min(n_cones, config.threads)
    blocks = cld(n_cones, threads)

    CUDA.@sync kernel(rowptr, rng_cones, row_shift, n_shift, n_cones; threads, blocks)
end

# #same as _kkt_count_diag, but counts places
# #where the input matrix M has a missing
# #diagonal entry.  M must be square and TRIU
# function _csr_colcount_missing_diag(K,M,initcol)

#     for i = 1:M.n
#         if((M.colptr[i] == M.colptr[i+1]) ||    #completely empty column
#            (M.rowval[M.colptr[i+1]-1] != i)     #last element is not on diagonal
#           )
#             K.colptr[i + (initcol-1)] += 1
#         end
#     end
# end

function _kernel_csr_rowcount_missing_diag_full(
    rowptr::AbstractVector{Cint}, 
    P::AbstractSparseMatrix{T},
    P_zero::AbstractVector{Cint}
) where {T}

    i = (blockIdx().x-one(Cint))*blockDim().x+threadIdx().x

    #just add the nonzero count each row
    if i <= P.dims[1]
        if(P.rowPtr[i] == P.rowPtr[i+1])    #completely empty row
            rowptr[i+1] += 1
            P_zero[i] = -1
            return nothing
        end
        @views colVali = P.colVal[P.rowPtr[i]:P.rowPtr[i+1]-1]     
        #check whether the i-th diagonal is missing
        @inbounds for idx in colVali
            if (idx >= i) 
                if (idx != i)
                    #i-th diagonal is missing
                    rowptr[i+1] += 1
                end
                P_zero[i] = idx
                return nothing
            end
        end
    end

    return nothing
end

@inline function _csr_rowcount_missing_diag_full_gpu(
    rowptr::AbstractVector{Cint}, 
    P::AbstractSparseMatrix{T},
    P_zero::AbstractVector{Cint}
) where {T}

    kernel = @cuda launch=false _kernel_csr_rowcount_missing_diag_full(rowptr, P, P_zero)
    config = launch_configuration(kernel.fun)
    threads = min(P.dims[1], config.threads)
    blocks = cld(P.dims[1], threads)

    CUDA.@sync kernel(rowptr, P, P_zero; threads, blocks)
end

# function _check_missing_diag_full(arr,k)
#     for val in arr
#         if (val >= k) 
#             return val  #val != k implies missing diagonal
#         end
#     end

#     return 0      #missing diagonal and all values in arr is smaller than k
# end

# #increment the K.colptr by the a number of nonzeros.
# #used to account for the placement of a column
# #vector that partially populates the column
# function _csr_colcount_colvec(K,n,firstrow, firstcol)

#     #just add the vector length to this column
#     K.colptr[firstcol] += n

# end

# #increment the K.colptr by 1 for every element
# #used to account for the placement of a column
# #vector that partially populates the column
# function _csr_colcount_rowvec(K,n,firstrow,firstcol)

#     #add one element to each of n consective columns
#     #starting from initcol.  The row index doesn't
#     #matter here.
#     for i = 1:n
#         K.colptr[firstcol + i - 1] += 1
#     end

# end

#increment the rowptr by the number of nonzeros in A 
function _kernel_csr_rowcount_block(
    rowptr::AbstractVector{Cint}, 
    A::AbstractSparseMatrix{T},
    shift::Cint
) where {T}

    i = (blockIdx().x-one(Cint))*blockDim().x+threadIdx().x

    #just add the nonzero count each row
    if i <= A.dims[1]
        rowptr[shift + i + 1] += A.rowPtr[i+1] - A.rowPtr[i]
    end

    return nothing
end

@inline function _csr_rowcount_block_gpu(
    rowptr::AbstractVector{Cint}, 
    A::AbstractSparseMatrix{T},
    shift::Cint
) where {T}
    #just add the row count
    kernel = @cuda launch=false _kernel_csr_rowcount_block(rowptr, A, shift)
    config = launch_configuration(kernel.fun)
    threads = min(A.dims[1], config.threads)
    blocks = cld(A.dims[1], threads)

    CUDA.@sync kernel(rowptr, A, shift; threads, blocks)
end

function _kernel_csr_rowcount_block_full(
    rowptr::AbstractVector{Cint}, 
    P::AbstractSparseMatrix{T},
    At::AbstractSparseMatrix{T}
) where {T}

    i = (blockIdx().x-one(Cint))*blockDim().x+threadIdx().x

    #just add the nonzero count each row
    if i <= P.dims[1]
        rowptr[i+1] += (P.rowPtr[i+1]-P.rowPtr[i]) + (At.rowPtr[i+1]-At.rowPtr[i])
    end

    return nothing
end

@inline function _csr_rowcount_block_full_gpu(
    rowptr::AbstractVector{Cint}, 
    P::AbstractSparseMatrix{T},
    At::AbstractSparseMatrix{T}
) where {T}

    @assert(P.dims[1] == At.dims[1]) #M and N should have the same column number

    #just add the nonzero count each row
    kernel = @cuda launch=false _kernel_csr_rowcount_block_full(rowptr, P, At)
    config = launch_configuration(kernel.fun)
    threads = min(P.dims[1], config.threads)
    blocks = cld(P.dims[1], threads)

    CUDA.@sync kernel(rowptr, P, At; threads, blocks)
end

#increment the rowptr by the number of nonzeros of the Hessian block
@inline function _csr_rowcount_Hessian_gpu(
    cones::CompositeConeGPU{T},
    rowptr::CuVector{Cint},
    m::Cint,
    n::Cint
) where {T}
    n_linear = cones.n_linear
    n_soc = cones.n_soc
    n_sparse_soc = cones.n_sparse_soc
    n_exp = cones.n_exp
    n_pow = cones.n_pow
    n_psd = cones.n_psd
    rng_cones = cones.rng_cones

    #Count the diagonal Hessian parts for zero, nonnegative and sparse second-order cones
    if (n_linear + n_sparse_soc) > 0
        CUDA.@allowscalar begin 
            rng = (one(Cint):rng_cones[n_linear+n_sparse_soc].stop) .+ (n + one(Cint)) 
            CUDA.@sync @views @. rowptr[rng] += 1
        end
    end
    
    #Count the additional sparse mapping for SOCs
    if n_sparse_soc > SPARSE_SOC_PARALELL_NUM
        _csr_count_sparse_soc_parallel(rowptr, rng_cones, m, n, n_linear, n_sparse_soc)
    else n_sparse_soc > 0
        # track the next sparse row to fill 
        prow = m + n + one(Cint) #next sparse row to fill
        n_sparse_soc = cones.n_sparse_soc
        CUDA.@allowscalar for i in (n_linear+one(Cint)):(n_linear+n_sparse_soc)

            #add sparse expansions rows for sparse socs  
            rng_cone_i = rng_cones[i] .+ (n + one(Cint))
            rowptr[prow+Cint(1)] = length(rng_cone_i) + 1
            rowptr[prow+Cint(2)] = length(rng_cone_i) + 1
            prow += Cint(2) #next sparse row to fill 

            #add sparse expansions columns for sparse cones 
            @views rowptr[rng_cone_i] .+= Cint(2)
        end
        synchronize()
    end

    #Count the remaining dense block
    n_rec = (n_soc - n_sparse_soc) + n_exp + n_pow + n_psd
    if (n_rec > 0)
        _csr_rowcount_dense_full_gpu(rowptr, rng_cones, n, n_linear+n_sparse_soc, n_rec)
    end
end

#increment the rowptr by the extended rows and columns of sparse socs in parallel
function _kernel_csr_count_sparse_soc_parallel(
    rowptr::AbstractVector{Cint}, 
    rng_cones::AbstractVector,
    m::Cint,
    n::Cint,
    n_linear::Cint,
    n_sparse_soc::Cint
)

    i = (blockIdx().x-one(Cint))*blockDim().x+threadIdx().x

    #just add the nonzero count each row
    if i <= n_sparse_soc
        #add sparse expansions rows for sparse socs  
        prow = m + n + Cint(2)*i
        rng_cone_i = rng_cones[n_linear + i] .+ (n + one(Cint))
        len_i = Cint(length(rng_cone_i))
        rowptr[prow] = len_i + 1
        rowptr[prow+one(Cint)] = len_i + 1

        #add sparse expansions columns for sparse cones 
        @inbounds for j in rng_cone_i
            rowptr[j] += Cint(2)
        end
    end

    return nothing
end

@inline function _csr_count_sparse_soc_parallel(
    rowptr::AbstractVector{Cint}, 
    rng_cones::AbstractVector,
    m::Cint,
    n::Cint,
    n_linear::Cint,
    n_sparse_soc::Cint
)
    #just add the row count
    kernel = @cuda launch=false _kernel_csr_count_sparse_soc_parallel(rowptr, rng_cones, m, n, n_linear, n_sparse_soc)
    config = launch_configuration(kernel.fun)
    threads = min(n_sparse_soc, config.threads)
    blocks = cld(n_sparse_soc, threads)

    CUDA.@sync kernel(rowptr, rng_cones, m, n, n_linear, n_sparse_soc; threads, blocks)
end


# #populate a partial column with zeros using the K.colptr as indicator of
# #next fill location in each row.
# function _csr_fill_colvec(K,vtoKKT,initrow,initcol)

#     for i = 1:length(vtoKKT)
#         dest               = K.colptr[initcol]
#         K.rowval[dest]     = initrow + i - 1
#         K.nzval[dest]      = 0.
#         vtoKKT[i]          = dest
#         K.colptr[initcol] += 1
#     end

# end

# #populate a partial row with zeros using the K.colptr as indicator of
# #next fill location in each row.
# function _csr_fill_rowvec(K,vtoKKT,initrow,initcol)

#     for i = 1:length(vtoKKT)
#         col            = initcol + i - 1
#         dest           = K.colptr[col]
#         K.rowval[dest] = initrow
#         K.nzval[dest]  = 0.
#         vtoKKT[i]      = dest
#         K.colptr[col] += 1
#     end

# end


#populate values from M using the rowptr as indicator of
#next fill location in each row.
function _kernel_csr_fill_block(
    rowptr::AbstractVector{Cint}, 
    colval::AbstractVector{Cint}, 
    nzval::AbstractVector{T}, 
    M::AbstractSparseMatrix{T}, 
    MtoKKT::AbstractVector{Cint}, 
    initrow::Cint, 
    initcol::Cint
) where {T}

    i = (blockIdx().x-one(Cint))*blockDim().x+threadIdx().x
    if i <= M.dims[1]
        @inbounds for j = M.rowPtr[i]:(M.rowPtr[i+1]-1)
            row = i + (initrow - 1)
            col = M.colVal[j] + (initcol - 1)

            dest           = rowptr[row]
            colval[dest] = col
            nzval[dest]  = M.nzVal[j]
            MtoKKT[j]      = dest
            rowptr[row] += 1
        end
    end

    return nothing
end

@inline function _csr_fill_block_gpu(
    rowptr::AbstractVector{Cint}, 
    colval::AbstractVector{Cint}, 
    nzval::AbstractVector{T}, 
    M::AbstractSparseMatrix{T}, 
    MtoKKT::AbstractVector{Cint}, 
    initrow::Cint, 
    initcol::Cint
) where {T}

    kernel = @cuda launch=false _kernel_csr_fill_block(rowptr, colval, nzval, M, MtoKKT, initrow, initcol)
    config = launch_configuration(kernel.fun)
    threads = min(M.dims[1], config.threads)
    blocks = cld(M.dims[1], threads)

    CUDA.@sync kernel(rowptr, colval, nzval, M, MtoKKT, initrow, initcol; threads, blocks)
end

function _csr_fill_specific_diag_full_gpu(
    rowptr::AbstractVector{Cint}, 
    colval::AbstractVector{Cint}, 
    nzval::AbstractVector{T}, 
    i::Cint
) where {T}
    dest           = rowptr[i]
    colval[dest] = i
    nzval[dest]  = 0        #explicitly fillin 0 values
    rowptr[i] += 1 
end

function _kernel_csr_fill_P_diag_full(
    rowptr::AbstractVector{Cint}, 
    colval::AbstractVector{Cint}, 
    nzval::AbstractVector{T}, 
    P::AbstractSparseMatrix{T}, 
    P_zero::AbstractVector{Cint}, 
    PtoKKT::AbstractVector{Cint}
) where {T}
    i = (blockIdx().x-one(Cint))*blockDim().x+threadIdx().x
    
    if i <= P.dims[1]
        if(P_zero[i] == -1)    #completely empty column
            dest           = rowptr[i]
            colval[dest] = i
            nzval[dest]  = 0        #explicitly fillin 0 values
            rowptr[i] += 1 
        else
            for j = P.rowPtr[i]:(P.rowPtr[i+1]-1)
                row = i
                col = P.colVal[j]
                if (col == P_zero[i] && col > row)            #insert the diagonal when it is missing in P and P has nonzero entry below the diagonal
                    dest           = rowptr[i]
                    colval[dest] = i
                    nzval[dest]  = 0        #explicitly fillin 0 values
                    rowptr[i] += 1  
                end
                dest           = rowptr[row]
                colval[dest] = col
                nzval[dest]  = P.nzVal[j]
                PtoKKT[j]      = dest
                rowptr[row] += 1
            end
            if (P_zero[i] == 0)            #insert the diagonal when it is missing in P and P has no nonzero entry below the diagonal
                dest           = rowptr[i]
                colval[dest] = i
                nzval[dest]  = 0        #explicitly fillin 0 values
                rowptr[i] += 1 
            end
        end
    end

    return nothing
end

#fill P block with missing diagonal
@inline function _csr_fill_P_diag_full_gpu(
    rowptr::AbstractVector{Cint}, 
    colval::AbstractVector{Cint}, 
    nzval::AbstractVector{T}, 
    P::AbstractSparseMatrix{T}, 
    P_zero::AbstractVector{Cint}, 
    PtoKKT::AbstractVector{Cint}
) where {T}
    kernel = @cuda launch=false _kernel_csr_fill_P_diag_full(rowptr, colval, nzval, P, P_zero, PtoKKT)
    config = launch_configuration(kernel.fun)
    threads = min(P.dims[1], config.threads)
    blocks = cld(P.dims[1], threads)

    CUDA.@sync kernel(rowptr, colval, nzval, P, P_zero, PtoKKT; threads, blocks)
end

#fill the colval of Hessian block in KKT
@inline function _csr_fill_Hessian_gpu(
    cones::CompositeConeGPU{T},
    map::GPUDataMap,
    rowptr::CuVector{Cint}, 
    colval::CuVector{Cint},
    nzval::CuVector{T},
    m::Cint,
    n::Cint
) where {T}
    rng_cones = cones.rng_cones
    rng_blocks = cones.rng_blocks
    n_linear = cones.n_linear
    n_sparse_soc = cones.n_sparse_soc

    #Fill the diagonal Hessian parts for zero, nonnegative and sparse second-order cones
    if (n_linear + n_sparse_soc) > 0
        CUDA.@allowscalar begin
            block = view(map.Hsblocks, one(Cint):(rng_blocks[n_linear+n_sparse_soc].stop))
            _csr_fill_diag_gpu(rowptr, colval, nzval, block, n + one(Cint), Cint(length(block)))
        end
        synchronize()
    end

    # Initializing additional rows (columns) for sparse second-order cones
    # track the next sparse row to fill
    if n_sparse_soc > SPARSE_SOC_PARALELL_NUM
        _csr_fill_sparse_soc_parallel(map.vu, map.vut, rowptr, colval, rng_cones, m, n, n_linear, n_sparse_soc, cones.numel_linear)
    else n_sparse_soc > 0
        _csr_fill_sparse_soc_sequential(map.vu, map.vut, rowptr, colval, rng_cones, m, n, n_linear, n_sparse_soc)
    end

    if n_sparse_soc > zero(Cint)
        #final settings for map.D 
        rng_D = (m + n + one(Cint)):(m + n + Cint(2)*n_sparse_soc)
        rowptr_D = view(rowptr, rng_D)
        CUDA.@sync @. map.D = rowptr_D
        _fill_range_gpu(map.D, colval, m+n)
        CUDA.@sync @. rowptr_D += one(Cint)

        #fill in nzval with 0s
        @views @. nzval[map.vu] = 0.
        @views @. nzval[map.vut] = 0.
        @views @. nzval[map.D] = 0.
    end
    synchronize()

    n_linear = cones.n_linear
    n_rec = (cones.n_soc - n_sparse_soc) + cones.n_exp + cones.n_pow + cones.n_psd
    if (n_rec > 0)
        _csr_fill_dense_full_gpu(rowptr, colval, nzval, map.Hsblocks, rng_cones, rng_blocks, n_linear+n_sparse_soc, n, n_rec)
    end
end

#fill the colval corresponding to the extended rows and columns of sparse socs in parallel
function _kernel_csr_fill_sparse_soc_parallel(
    mapvu::AbstractVector{Cint},
    mapvut::AbstractVector{Cint},
    rowptr::AbstractVector{Cint}, 
    colval::AbstractVector{Cint},
    rng_cones::AbstractVector,
    m::Cint,
    n::Cint,
    n_linear::Cint,
    n_sparse_soc::Cint,
    numel_shift::Cint
)

    i = (blockIdx().x-one(Cint))*blockDim().x+threadIdx().x

    #just add the nonzero count each row
    if i <= n_sparse_soc
        prow = m + n + Cint(2)*i - one(Cint)
        #add sparse expansions columns for sparse cones 
        rng_cone_i = rng_cones[n_linear + i]
        rng_row_i = rng_cone_i .+ n
        len_i = Cint(length(rng_row_i))
        shift = Cint(2)*(rng_cone_i.start - numel_shift - one(Cint))
        rng_vec_vt = (shift+one(Cint)):(shift+len_i)
        rng_vec_ut = (shift+len_i+one(Cint)):(shift+Cint(2)*len_i)

        rowptr_i = view(rowptr, rng_row_i)
        vi = view(mapvu, rng_vec_vt)
        ui = view(mapvu, rng_vec_ut)

        #add extended columns for sparse cones 
        @inbounds for j in one(Cint):len_i
            vi[j] = rowptr_i[j]
            colval[rowptr_i[j]] = prow
            rowptr_i[j] += one(Cint)

            ui[j] = rowptr_i[j] 
            colval[rowptr_i[j]] = prow + one(Cint)
            rowptr_i[j] += one(Cint)
        end

        #add extended rows for sparse cones 
        start_col = rng_row_i.start - one(Cint)

        start_v = rowptr[prow] - one(Cint)
        start_u = rowptr[prow+one(Cint)] - one(Cint)
        vti = view(mapvut, rng_vec_vt)
        uti = view(mapvut, rng_vec_ut)

        @inbounds for j in one(Cint):len_i
            vti[j] = start_v + j
            colval[start_v + j] = start_col + j
        end
        @inbounds for j in one(Cint):len_i
            uti[j] = start_u + j
            colval[start_u + j] = start_col + j
        end
        
        rowptr[prow] += len_i
        rowptr[prow+one(Cint)] += len_i

    end

    return nothing
end

@inline function _csr_fill_sparse_soc_parallel(
    mapvu::AbstractVector{Cint},
    mapvut::AbstractVector{Cint},
    rowptr::AbstractVector{Cint}, 
    colval::AbstractVector{Cint},
    rng_cones::AbstractVector,
    m::Cint,
    n::Cint,
    n_linear::Cint,
    n_sparse_soc::Cint,
    numel_shift::Cint
)
    #just add the row count
    kernel = @cuda launch=false _kernel_csr_fill_sparse_soc_parallel(mapvu, mapvut, rowptr, colval, rng_cones, m, n, n_linear, n_sparse_soc, numel_shift)
    config = launch_configuration(kernel.fun)
    threads = min(n_sparse_soc, config.threads)
    blocks = cld(n_sparse_soc, threads)

    CUDA.@sync kernel(mapvu, mapvut, rowptr, colval, rng_cones, m, n, n_linear, n_sparse_soc, numel_shift; threads, blocks)
end

@inline function _csr_fill_sparse_soc_sequential(
    mapvu::AbstractVector{Cint},
    mapvut::AbstractVector{Cint},
    rowptr::AbstractVector{Cint}, 
    colval::AbstractVector{Cint},
    rng_cones::AbstractVector,
    m::Cint,
    n::Cint,
    n_linear::Cint,
    n_sparse_soc::Cint
)
    prow = m + n + one(Cint) #next sparse row to fill
    sparse_idx = zero(Cint)

    CUDA.@allowscalar for i in one(Cint):n_sparse_soc
        #add sparse expansions columns for sparse cones 
        rng_cone_i = rng_cones[n_linear + i] .+ n
        len_i = Cint(length(rng_cone_i))
        rng_vec_vt = (sparse_idx+one(Cint)):(sparse_idx+len_i)
        rng_vec_ut = (sparse_idx+len_i+one(Cint)):(sparse_idx+Cint(2)*len_i)

        rowptr_i = view(rowptr, rng_cone_i)
        #regard each vi ui sequentially
        @views @. mapvu[rng_vec_vt] = rowptr_i
        @views @. colval[rowptr_i] = prow
        @views @. rowptr_i += one(Cint)
        synchronize()

        @views @. mapvu[rng_vec_ut] = rowptr_i
        @views @. colval[rowptr_i] = prow + one(Cint)
        @views @. rowptr_i += one(Cint)
        synchronize()

        #add sparse expansions rows for sparse cones 
        start_col = rng_cone_i.start - one(Cint)

        start_v = rowptr[prow] - one(Cint)
        start_u = rowptr[prow+one(Cint)] - one(Cint)
        vti = view(mapvut, rng_vec_vt)
        uti = view(mapvut, rng_vec_ut)
        _csr_fill_sparse_uv_gpu(vti, uti, colval, start_col, start_v, start_u)
        rowptr[prow] += len_i
        rowptr[prow+one(Cint)] += len_i

        sparse_idx += Cint(2)*len_i
        prow += Cint(2) #next sparse soc to fill 
    end
end

# #Populate the upper or lower triangle with 0s using the K.colptr
# #as indicator of next fill location in each row
# function _csr_fill_dense_triangle(K,blocktoKKT,offset,blockdim,shape)

#     #data will always be supplied as triu, so when filling it into
#     #a tril shape we also need to transpose it.   Just write two
#     #separate cases for clarity here

#     if(shape === :triu)
#         _fill_dense_triangle_triu(K,blocktoKKT,offset,blockdim)
#     else #shape ==== :tril
#         _fill_dense_triangle_tril(K,blocktoKKT,offset,blockdim)
#     end
# end

# function _fill_dense_triangle_triu(K,blocktoKKT,offset,blockdim)

#     kidx = 1
#     for col in offset:(offset + blockdim - 1)
#         for row in (offset:col)
#             dest             = K.colptr[col]
#             K.rowval[dest]   = row
#             K.nzval[dest]    = 0.  #structural zero
#             K.colptr[col]   += 1
#             blocktoKKT[kidx] = dest
#             kidx = kidx + 1
#         end
#     end
# end

# function _fill_dense_triangle_tril(K,blocktoKKT,offset,blockdim)

#     kidx = 1
#     for row in offset:(offset + blockdim - 1)
#         for col in offset:row
#             dest             = K.colptr[col]
#             K.rowval[dest]   = row
#             K.nzval[dest]    = 0.  #structural zero
#             K.colptr[col]   += 1
#             blocktoKKT[kidx] = dest
#             kidx = kidx + 1
#         end
#     end
# end

function _kernel_csr_fill_dense_full(
    rowptr::AbstractVector{Cint}, 
    colval::AbstractVector{Cint}, 
    nzval::AbstractVector{T}, 
    Hsblocks::AbstractVector{Cint},
    rng_cones::AbstractVector,
    rng_blocks::AbstractVector,
    cone_shift::Cint,
    n::Cint,
    n_rec::Cint
) where {T}
    i = (blockIdx().x-one(Cint))*blockDim().x+threadIdx().x
    
    if i <= n_rec
        shift_i = cone_shift+i
        @views block_i = Hsblocks[rng_blocks[shift_i]]
        rng_cone_i = rng_cones[shift_i] .+ n
        idx = 1
        @inbounds for row in rng_cone_i
            @inbounds for col in rng_cone_i
                dest           = rowptr[row]
                colval[dest]   = col
                nzval[dest]    = 0  #structural zero
                rowptr[row]   += 1
                block_i[idx] = dest
                idx = idx + 1
            end
        end
    end

    return nothing
end

@inline function _csr_fill_dense_full_gpu(
    rowptr::AbstractVector{Cint}, 
    colval::AbstractVector{Cint}, 
    nzval::AbstractVector{T}, 
    Hsblocks::AbstractVector{Cint},
    rng_cones::AbstractVector,
    rng_blocks::AbstractVector,
    cone_shift::Cint,
    n::Cint,
    n_rec::Cint
) where {T}
    kernel = @cuda launch=false _kernel_csr_fill_dense_full(rowptr, colval, nzval, Hsblocks, rng_cones, rng_blocks, cone_shift, n, n_rec)
    config = launch_configuration(kernel.fun)
    threads = min(n_rec, config.threads)
    blocks = cld(n_rec, threads)

    CUDA.@sync kernel(rowptr, colval, nzval, Hsblocks, rng_cones, rng_blocks, cone_shift, n, n_rec; threads, blocks)
end

#Populate the diagonal with 0s using the K.colptr as indicator of
#next fill location in each row
function _kernel_csr_fill_diag(
    rowptr::AbstractVector{Cint}, 
    colval::AbstractVector{Cint}, 
    nzval::AbstractVector{T},
    diagtoKKT::AbstractVector{Cint},
    offset::Cint,
    blockdim::Cint
) where {T}

    i = (blockIdx().x-one(Cint))*blockDim().x+threadIdx().x
    if i <= blockdim
        row                 = i + offset - 1
        dest                = rowptr[row]
        colval[dest]      = row
        nzval[dest]       = 0.  #structural zero
        rowptr[row]      += 1
        diagtoKKT[i]        = dest
    end

    return nothing
end

@inline function _csr_fill_diag_gpu(
    rowptr::AbstractVector{Cint}, 
    colval::AbstractVector{Cint}, 
    nzval::AbstractVector{T},
    diagtoKKT::AbstractVector{Cint},
    offset::Cint,
    blockdim::Cint
) where {T}

    kernel = @cuda launch=false _kernel_csr_fill_diag(rowptr, colval, nzval, diagtoKKT, offset, blockdim)
    config = launch_configuration(kernel.fun)
    threads = min(blockdim, config.threads)
    blocks = cld(blockdim, threads)

    CUDA.@sync kernel(rowptr, colval, nzval, diagtoKKT, offset, blockdim; threads, blocks)
end

# Populate indices for ut, vt of a sparse second-order cones
function _kernel_csr_fill_sparsecone(
    vt::AbstractVector{Cint}, 
    ut::AbstractVector{Cint}, 
    colval::AbstractVector{Cint}, 
    start_col::Cint, 
    start_v::Cint,
    start_u::Cint,
    blockdim::Cint
)

    i = (blockIdx().x-one(Cint))*blockDim().x+threadIdx().x
    if i <= blockdim
        vt[i] = start_v + i
        ut[i] = start_u + i
        colval[vt[i]] = start_col + i
        colval[ut[i]] = start_col + i
    end

    return nothing
end

@inline function _csr_fill_sparse_uv_gpu(
    vt::AbstractVector{Cint}, 
    ut::AbstractVector{Cint}, 
    colval::AbstractVector{Cint}, 
    start_col::Cint, 
    start_v::Cint,
    start_u::Cint
)
    blockdim = Cint(length(vt))
    kernel = @cuda launch=false _kernel_csr_fill_sparsecone(vt, ut, colval, start_col, start_v, start_u, blockdim)
    config = launch_configuration(kernel.fun)
    threads = min(blockdim, config.threads)
    blocks = cld(blockdim, threads)

    CUDA.@sync kernel(vt, ut, colval, start_col, start_v, start_u, blockdim; threads, blocks)
end

function _kernel_fill_range!(
    rowptr::AbstractVector{Cint},
    colval::AbstractVector{Cint},
    shift::Cint,
    blockdim::Cint
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= blockdim
        colval[rowptr[i]] = shift + i
    end
    return
end

@inline function _fill_range_gpu(
    rowptr::AbstractVector{Cint},
    colval::AbstractVector{Cint},
    shift::Cint
)   
    blockdim = Cint(length(rowptr))
    kernel = @cuda launch=false _kernel_fill_range!(rowptr, colval, shift, blockdim)
    config = launch_configuration(kernel.fun)
    threads = min(blockdim, config.threads)
    blocks = cld(blockdim, threads)

    CUDA.@sync kernel(rowptr, colval, shift, blockdim; threads, blocks)
end

# #same as _csr_fill_diag, but only places 0.
# #entries where the input matrix M has a missing
# #diagonal entry.  M must be square and TRIU
# function _csr_fill_missing_diag(K,M,initcol)

#     for i = 1:M.n
#         #fill out missing diagonal terms only
#         if((M.colptr[i] == M.colptr[i+1]) ||    #completely empty column
#            (M.rowval[M.colptr[i+1]-1] != i)     #last element is not on diagonal
#           )
#             dest           = K.colptr[i + (initcol - 1)]
#             K.rowval[dest] = i + (initcol - 1)
#             K.nzval[dest]  = 0.  #structural zero
#             K.colptr[i]   += 1
#         end
#     end
# end

function _csr_rowcount_to_rowptr_gpu(
    rowptr::AbstractVector{Cint}
)
    CUDA.@allowscalar rowptr[1] = 1

    #Efficient scan operations for the cumulative sum
    CUDA.accumulate!(+, rowptr, rowptr)
 
end

function _kkt_backshift_colptrs_gpu(rowptr::AbstractVector{Cint})

    rowptr_tmp = deepcopy(rowptr)
    @views copyto!(rowptr[2:end], rowptr_tmp[1:end-1])
    CUDA.@allowscalar rowptr[1] = 1  #zero in C
    CUDA.unsafe_free!(rowptr_tmp)                #release the memory

end


function _kernel_map_diag_full(
    rowptr::AbstractVector{Cint}, 
    colval::AbstractVector{Cint}, 
    diagind::AbstractVector{Cint}
)
    i = (blockIdx().x-one(Cint))*blockDim().x+threadIdx().x
    
    if i <= length(diagind)
        @inbounds for j = rowptr[i]:(rowptr[i+1]-1)
            if (colval[j] == i) 
                diagind[i] = j
                return nothing
            end
        end
    end

    return nothing
end

@inline function _map_diag_full_gpu(
    rowptr::AbstractVector{Cint}, 
    colval::AbstractVector{Cint}, 
    diagind::AbstractVector{Cint}
)
    n = length(diagind)
    kernel = @cuda launch=false _kernel_map_diag_full(rowptr, colval, diagind)
    config = launch_configuration(kernel.fun)
    threads = min(n, config.threads)
    blocks = cld(n, threads)

    CUDA.@sync kernel(rowptr, colval, diagind; threads, blocks)
end

# function _count_diagonal_entries(P)

#     count = 0
#     for i = 1:P.n

#         #compare last entry in each column with
#         #its row number to identify diagonal entries
#         if((P.colptr[i+1] != P.colptr[i]) &&    #nonempty column
#            (P.rowval[P.colptr[i+1]-1] == i) )   #last element is on diagonal
#                 count += 1
#         end
#     end
#     return count

# end

function _kernel_count_diagonal_entries_full!(P::AbstractSparseMatrix{T}, count::AbstractVector{Bool}) where {T <: AbstractFloat}

	n = size(P,1)

    i = (blockIdx().x-one(Cint))*blockDim().x+threadIdx().x
	if i <= n
        @inbounds for j = P.rowPtr[i]:P.rowPtr[i+1] -1
            if (P.colVal[j] == i) 
                count[i] = true 
            end
        end
	end
	return nothing
end

function _count_diagonal_entries_full_gpu(P::AbstractSparseMatrix)

    m, n = size(P)
    @assert(m == n)
    count = CuVector{Bool}(undef, m)
    CUDA.fill!(count, false)
    kernel = @cuda launch=false _kernel_count_diagonal_entries_full!(P, count)
    config = launch_configuration(kernel.fun)
    threads = min(m, config.threads)
    blocks = cld(m, threads)

    CUDA.@sync kernel(P, count; threads, blocks)
    total_count = sum(count)
    CUDA.unsafe_free!(count)        #release the memory of unused memory

    return total_count

end