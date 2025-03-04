import CUDA.CUSPARSE: CuSparseDeviceMatrixCSR
import CUDA.CUBLAS: unsafe_strided_batch, handle
import CUDA.CUBLAS: cublasStatus_t, cublasHandle_t, cublasFillMode_t
import CUDA.CUSOLVER: cusolverDnHandle_t, cusolverStatus_t
import CUDA: unsafe_free!
using LinearAlgebra.LAPACK: chkargsok, chklapackerror, chktrans, chkside, chkdiag, chkuplo
# using Libdl

#############################################
#  converts an elementwise scaling into
# a scaling that preserves cone memership
#############################################
function _kernel_rectify_equilibration!(
    δ::AbstractVector{T},
    e::AbstractVector{T},
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_rec::Cint
 ) where{T}
 
     i = (blockIdx().x-1)*blockDim().x+threadIdx().x
 
     if i <= n_rec
         shift_i = i + n_shift
         rng_cone_i = rng_cones[shift_i]
         @views δi = δ[rng_cone_i] 
         @views ei = e[rng_cone_i] 

     
        #all cones default to scalar equilibration
        #unless otherwise specified
        tmp    = mean(ei)
        @inbounds for j in 1:length(rng_cone_i)
            δi[j]    = tmp / ei[j]
        end
     end
 
     return nothing
 end

@inline function rectify_equilibration_gpu!(
    δ::AbstractVector{T},
    e::AbstractVector{T},
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_rec::Cint
 ) where{T}
 
    kernel = @cuda launch=false _kernel_rectify_equilibration!(δ, e, rng_cones, n_shift, n_rec)
    config = launch_configuration(kernel.fun)
    threads = min(n_rec, config.threads)
    blocks = cld(n_rec, threads)

    CUDA.@sync kernel(δ, e, rng_cones, n_shift, n_rec; threads, blocks)
 end


 ############################################
 # Operations for equilibration
 ############################################
 @inline function kkt_row_norms_gpu!(
    P::AbstractSparseMatrix{T},
    A::AbstractSparseMatrix{T},
    At::AbstractSparseMatrix{T},
    norm_LHS::AbstractVector{T},
    norm_RHS::AbstractVector{T}
) where {T}

    row_norms_gpu!(norm_LHS, P)   # P is a full CSR matrix on GPU
	row_norms_no_reset_gpu!(norm_LHS, At)       #incrementally from P norms
	row_norms_gpu!(norm_RHS, A)                      #A is a CSR matrix on GPU

	return nothing
end

function _kernel_row_norms!(
    norms::AbstractVector{T},
	A::AbstractSparseMatrix{T}
 ) where{T}
 
     i = (blockIdx().x-1)*blockDim().x+threadIdx().x
 
     if i <= length(norms)
        @inbounds for j = A.rowPtr[i]:(A.rowPtr[i + 1] - 1)
			tmp = abs(A.nzVal[j])
			norms[i] = norms[i] > tmp ? norms[i] : tmp
		end
     end
 
     return nothing
 end

@inline function row_norms_gpu!(
    norms::AbstractVector{T},
	A::AbstractSparseMatrix{T}
) where{T <: AbstractFloat}

    fill!(norms, zero(T))
    return row_norms_no_reset_gpu!(norms,A)
end

@inline function row_norms_no_reset_gpu!(
    norms::AbstractVector{T},
	A::AbstractSparseMatrix{T};
	reset::Bool = true
) where{T <: AbstractFloat}
    n = length(norms)

    kernel = @cuda launch=false _kernel_row_norms!(norms, A)
    config = launch_configuration(kernel.fun)
    threads = min(n, config.threads)
    blocks = cld(n, threads)

    CUDA.@sync kernel(norms, A; threads, blocks)
	return nothing
end

@inline function scalarmul_gpu!(A::AbstractSparseMatrix{T}, c::T) where {T}
	CUDA.@sync @. A.nzVal *= c
end

function _kernel_lrscale_gpu!(L::AbstractVector{T}, M::AbstractSparseMatrix{T}, R::AbstractVector{T}) where {T <: AbstractFloat}

	m, n = size(M)
	Mnzval  = M.nzVal
	Mrowptr = M.rowPtr
	Mcolval = M.colVal

    i = (blockIdx().x-1)*blockDim().x+threadIdx().x
	if i <= m
		for j = Mrowptr[i]:(Mrowptr[i + 1] - 1)
	 		Mnzval[j] *= R[Mcolval[j]] * L[i]
		end
	end
	return nothing
end

@inline function lrscale_gpu!(L::AbstractVector{T}, M::AbstractSparseMatrix{T}, R::AbstractVector{T}) where {T <: AbstractFloat}

	m, n = size(M)
	(m == length(L) && n == length(R)) || throw(DimensionMismatch())

    kernel = @cuda launch=false _kernel_lrscale_gpu!(L, M, R)
    config = launch_configuration(kernel.fun)
    threads = min(m, config.threads)
    blocks = cld(m, threads)

    CUDA.@sync kernel(L, M, R; threads, blocks)
	return nothing
end

function _kernel_lscale_gpu!(L::AbstractVector{T}, M::AbstractSparseMatrix{T}) where {T <: AbstractFloat}

	m, n = size(M)
	Mnzval  = M.nzVal
	Mrowptr = M.rowPtr

    i = (blockIdx().x-1)*blockDim().x+threadIdx().x
	if i <= m
		for j = Mrowptr[i]:(Mrowptr[i + 1] - 1)
	 		Mnzval[j] *= L[i]
		end
	end
	return nothing
end

@inline function lscale_gpu!(L::AbstractVector{T}, M::AbstractSparseMatrix{T}) where {T <: AbstractFloat}

	#NB : Same as:  @views M.nzval .*= L[M.rowval]
	#but this way allocates no memory at all and
	#is marginally faster
	m, n = size(M)
	(m == length(L)) || throw(DimensionMismatch())

    kernel = @cuda launch=false _kernel_lscale_gpu!(L, M)
    config = launch_configuration(kernel.fun)
    threads = min(m, config.threads)
    blocks = cld(m, threads)

    CUDA.@sync kernel(L, M; threads, blocks)
end

function _kernel_rscale_gpu!(M::AbstractSparseMatrix{T}, R::AbstractVector{T}) where {T <: AbstractFloat}

	m, n = size(M)
	Mnzval  = M.nzVal
	Mrowptr = M.rowPtr
    Mcolval = M.colVal

    i = (blockIdx().x-1)*blockDim().x+threadIdx().x
	if i <= m
		for j = Mrowptr[i]:(Mrowptr[i + 1] - 1)
	 		Mnzval[j] *= R[Mcolval[j]]
		end
	end
	return nothing
end

@inline function rscale_gpu!(M::AbstractSparseMatrix{T}, R::AbstractVector{T}) where {T <: AbstractFloat}

	#NB : Same as:  @views M.nzval .*= L[M.rowval]
	#but this way allocates no memory at all and
	#is marginally faster
	m, n = size(M)
	(n == length(R)) || throw(DimensionMismatch())

    kernel = @cuda launch=false _kernel_rscale_gpu!(M, R)
    config = launch_configuration(kernel.fun)
    threads = min(m, config.threads)
    blocks = cld(m, threads)

    CUDA.@sync kernel(M, R; threads, blocks)
end

#############################################
# dot operator
# NT: can be possibly optimized when using the shared memory
#############################################
function _kernel_dot_shifted_gpu(
    work::AbstractVector{T},
    z::AbstractVector{T}, 
    s::AbstractVector{T},
    dz::AbstractVector{T}, 
    ds::AbstractVector{T},
    α::T
) where {T<:Real}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= length(z)
        zi = z[i] + α * dz[i]
        si = s[i] + α * ds[i]
        work[i] = zi * si
    end

    return nothing
end

@inline function dot_shifted_gpu(
    work::AbstractVector{T},
    z::AbstractVector{T}, 
    s::AbstractVector{T},
    dz::AbstractVector{T}, 
    ds::AbstractVector{T},
    α::T
) where {T<:Real}
    
    n = length(work)
    kernel = @cuda launch=false _kernel_dot_shifted_gpu(work, z, s, dz, ds, α)
    config = launch_configuration(kernel.fun)
    threads = min(n, config.threads)
    blocks = cld(n, threads)
    
    CUDA.@sync kernel(work, z, s, dz, ds, α; threads, blocks)
    
    return sum(work)
end

#############################################
# Batched Cholesky (in-place decomposition)
#############################################
# potrfBatched - performs Cholesky factorizations
#Float64
function cusolverDnDpotrfBatched(handle, uplo, n, A, lda, info, batchSize)
    ccall((:cusolverDnDpotrfBatched, CUDA.libcusolver), cusolverStatus_t, 
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Ptr{Cdouble}}, Cint,
                   CuPtr{Cint}, Cint),
                   handle, uplo, n, A, lda, info, batchSize)
end

#Float32
function cusolverDnSpotrfBatched(handle, uplo, n, A, lda, info, batchSize)
    ccall((:cusolverDnSpotrfBatched, CUDA.libcusolver), cusolverStatus_t, 
                   (cusolverDnHandle_t, cublasFillMode_t, Cint, CuPtr{Ptr{Cdouble}}, Cint,
                   CuPtr{Cint}, Cint),
                   handle, uplo, n, A, lda, info, batchSize)
end
for (fname, elty) in ((:cusolverDnSpotrfBatched, :Float32),
    (:cusolverDnDpotrfBatched, :Float64)
    )
    @eval begin
        function potrfBatched!(A::CuArray{$elty, 3},uplo::Char)

            # Set up information for the solver arguments
            chkuplo(uplo)
            n = LinearAlgebra.checksquare(A[:,:,1])
            lda = max(1, stride(A[:,:,1], 2))
            batchSize = size(A,3)

            Aptrs = unsafe_strided_batch(A)

            dh = CUDA.CUSOLVER.dense_handle()
            resize!(dh.info, batchSize)

            # Run the solver
            $fname(dh, uplo, n, Aptrs, lda, dh.info, batchSize)

            # Copy the solver info and delete the device memory
            info = CUDA.@allowscalar collect(dh.info)

            # Double check the solver's exit status
            for i = 1:batchSize
                chkargsok(CUDA.CUSOLVER.BlasInt(info[i]))
            end

            # info[i] > 0 means the leading minor of order info[i] is not positive definite
            # LinearAlgebra.LAPACK does not throw Exception here
            # to simplify calls to isposdef! and factorize
            return A, info
        end
    end
end

#########################################################
# Masked zeros
#########################################################
function _kernel_mask_zeros!(
    A::AbstractArray{T, 3},
    uplo::Char,
    n::Clong
) where {T}

    i = (blockIdx().x-1)*blockDim().x+threadIdx().x

    if i <= n
        @views Ai = A[:,:,i]
        tril!(Ai)
    end

    return nothing
end

@inline function mask_zeros!(
    A::AbstractArray{T, 3},
    uplo::Char
) where {T}
    n = size(A,3)

    kernel = @cuda launch=false _kernel_mask_zeros!(A, uplo, n)
    config = launch_configuration(kernel.fun)
    threads = min(n, config.threads)
    blocks = cld(n, threads)

    CUDA.@sync kernel(A, uplo, n; threads, blocks)
end

#########################################################
# lrscale for psd cones
#########################################################
function _kernel_lrscale!(
    A::AbstractArray{T, 3},
    d::AbstractMatrix{T},
    n::Clong
) where {T}

    i = (blockIdx().x-1)*blockDim().x+threadIdx().x

    if i <= n
        @views Ai = A[:,:,i]
        @views di = d[:,i]
        lrscale!(di,Ai,di)
    end

    return nothing
end

@inline function lrscale_psd!(
    A::AbstractArray{T,3},
    d::AbstractMatrix{T}
) where {T}
    n = size(A,3)

    kernel = @cuda launch=false _kernel_lrscale!(A, d, n)
    config = launch_configuration(kernel.fun)
    threads = min(n, config.threads)
    blocks = cld(n, threads)

    CUDA.@sync kernel(A, d, n; threads, blocks)
    
end

#########################################################
# symmetric_part
#########################################################
function _kernel_symmetric_part!(
    A::AbstractArray{T, 3},
    n::Clong
) where {T}

    i = (blockIdx().x-1)*blockDim().x+threadIdx().x

    if i <= n
        @views Ai = A[:,:,i]
        symmetric_part!(Ai)
    end

    return nothing
end

@inline function symmetric_part_gpu!(
    A::AbstractArray{T,3}
) where {T}
    n = size(A,3)

    kernel = @cuda launch=false _kernel_symmetric_part!(A, n)
    config = launch_configuration(kernel.fun)
    threads = min(n, config.threads)
    blocks = cld(n, threads)

    CUDA.@sync kernel(A, n; threads, blocks)
    
end

function norm_scaled_gpu(m::AbstractVector{T},v::AbstractVector{T},work::AbstractVector{T}) where{T}
    CUDA.@sync @. work = m*v
    CUDA.@sync @. work = work*work
    t = sum(work)
    return sqrt(t)
end