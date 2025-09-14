# -------------------------------------
# KKTSolver using cudss direct solvers
# -------------------------------------

const CuVectorView{T} = SubArray{T, 1, CuVector{T}, Tuple{CuVector{Int}}, false}
##############################################################
# YC: Some functions are repeated as in the direct solver, which are better to be removed
##############################################################
mutable struct GPULDLKKTSolver{T} <: AbstractKKTSolver{T}

    # problem dimensions
    m::Cint; n::Cint; p::Cint;

    # Left and right hand sides for solves
    x::CuVector{T}
    b::CuVector{T}

    # internal workspace for IR scheme
    # and static offsetting of KKT
    work1::CuVector{T}
    work2::CuVector{T}

    #KKT mapping from problem data to KKT
    map::GPUDataMap 

    cones::CompositeConeGPU{T}
    P::CuSparseMatrix{T}
    A::CuSparseMatrix{T}
    At::CuSparseMatrix{T}

    #the expected signs of D in KKT = LDL^T
    Dsigns::CuVector{Cint}

    # a vector for storing the Hs blocks
    # on the in the KKT matrix block diagonal
    Hsblocks::CuVector{T}

    #unpermuted KKT matrix
    KKT::CuSparseMatrix{T}

    #settings just points back to the main solver settings.
    #Required since there is no separate LDL settings container
    settings::Settings{T}

    #the direct linear LDL solver
    GPUsolver::AbstractDirectLDLSolver{T}

    #the diagonal regularizer currently applied
    diagonal_regularizer::T

    function GPULDLKKTSolver{T}(
        P::CuSparseMatrix{T},
        A::CuSparseMatrix{T},
        At::CuSparseMatrix{T},
        cones::CompositeConeGPU{T},
        m::Cint,
        n::Cint,
        settings::Settings{T}
    ) where {T}

        # get a constructor for the LDL solver we should use,
        # and also the matrix shape it requires
        (kktshape, GPUsolverT) = get_ldlsolver_config(settings)

        #construct a KKT matrix of the right shape
        KKT, map = _assemble_full_kkt_matrix(P,A,At,cones)

        #Need this many extra variables for sparse cones
        p = length(map.D)

        #updates to the diagonal of KKT will be
        #assigned here before updating matrix entries
        dim = m + n + p

        #LHS/RHS/work for iterative refinement
        x    = CuVector{T}(undef,dim)
        b    = CuVector{T}(undef,dim)
        work1 = CuVector{T}(undef,dim)
        work2 = CuVector{T}(undef,dim)

        #the expected signs of D in LDL
        Dsigns = CUDA.ones(Cint,dim)
        @views fill!(Dsigns[n+1:n+m], -one(Cint))
        @views fill!(Dsigns[n+m+1:2:n+m+p], -one(Cint))     #for sparse SOCs

        Hsblocks = _allocate_kkt_Hsblocks_gpu(T, cones)
        diagonal_regularizer = zero(T)

        #the indirect linear solver engine
        GPUsolver = GPUsolverT{T}(KKT,x,b)

        return new(m,n,p,x,b,
                   work1,work2,map,
                   cones,P,A,At,
                   Dsigns,
                   Hsblocks,
                   KKT,settings,GPUsolver,
                   diagonal_regularizer
                   )
    end

end

GPULDLKKTSolver(args...) = GPULDLKKTSolver{DefaultFloat}(args...)

#update entries in the kktsolver object using the
#given index into its CSC representation
function _update_values!(
    GPUsolver::AbstractDirectLDLSolver{T},
    KKT::CuSparseMatrix{T},
    index::CuVector{Ti},
    values::CuVector{T}
) where{T,Ti}

    #Update values in the KKT matrix K
    CUDA.@sync @. KKT.nzVal[index] = values

end

@inline function _update_value!(
    KKT::AbstractMatrix{T},
    index::Ti,
    value::T
) where{T,Ti}

    #Update values in the KKT matrix K
    KKT.nzVal[index] = value

end

@inline function _update_value!(
    nzVal::AbstractVector{T},
    index::Ti,
    value::T
) where{T,Ti}

    #Update values in the KKT matrix K
    nzVal[index] = value

end

function kktsolver_linear_solver_info(
    kktsolver::GPULDLKKTSolver{T}
) where {T}
    linear_solver_info(kktsolver.GPUsolver)
end

#updates KKT matrix values
function _update_diag_values_KKT!(
    KKT::CuSparseMatrix{T},
    index::CuVector{Ti},
    values::CuVector{T}
) where{T,Ti}

    #Update values in the KKT matrix K
    @views copyto!(KKT.nzVal[index], values)
    
end

function kktsolver_update!(
    kktsolver:: GPULDLKKTSolver{T},
    cones::CompositeConeGPU{T}
) where {T}

    # the internal  GPUsolver is type unstable, so multiple
    # calls to the  GPUsolvers will be very slow if called
    # directly.   Grab it here and then call an inner function
    # so that the  GPUsolver has concrete type
    GPUsolver = kktsolver.GPUsolver
    return _kktsolver_update_inner!(kktsolver,GPUsolver,cones)
end


function _kktsolver_update_inner!(
    kktsolver:: GPULDLKKTSolver{T},
    GPUsolver::AbstractDirectLDLSolver{T},
    cones::CompositeConeGPU{T}
) where {T}

    #real implementation is here, and now  GPUsolver
    #will be compiled to something concrete.

    map       = kktsolver.map
    KKT       = kktsolver.KKT

    #Set the elements the W^tW blocks in the KKT matrix.
    get_Hs!(cones,kktsolver.Hsblocks)

    CUDA.@sync @. kktsolver.Hsblocks *= -one(T)
    _update_values!(GPUsolver,KKT,map.Hsblocks,kktsolver.Hsblocks)

    # Update for the KKT part of the sparse socs
    _update_KKT_sparse_expandable(GPUsolver, KKT, map, cones)

    return _kktsolver_regularize_and_refactor!(kktsolver, GPUsolver)

end

function _kktsolver_regularize_and_refactor!(
    kktsolver::GPULDLKKTSolver{T},
    GPUsolver::AbstractDirectLDLSolver{T}
) where{T}

    settings      = kktsolver.settings
    map           = kktsolver.map
    KKT        = kktsolver.KKT
    Dsigns        = kktsolver.Dsigns
    diag_kkt      = kktsolver.work1
    diag_shifted  = kktsolver.work2


    if(settings.static_regularization_enable)

        # hold a copy of the true KKT diagonal
        CUDA.@sync @views diag_kkt .= KKT.nzVal[map.diag_full]
        ϵ = _compute_regularizer(diag_kkt, settings)

        # compute an offset version, accounting for signs
        CUDA.@sync @. diag_shifted = diag_kkt + Dsigns*ϵ

        # overwrite the diagonal of KKT and within the  GPUsolver
        _update_diag_values_KKT!(KKT,map.diag_full,diag_shifted)

        # remember the value we used.  Not needed,
        # but possibly useful for debugging
        kktsolver.diagonal_regularizer = ϵ

    end

    is_success = refactor!(GPUsolver)

    if(settings.static_regularization_enable)

        # put our internal copy of the KKT matrix back the way
        # it was. Not necessary to fix the  GPUsolver copy because
        # this is only needed for our post-factorization IR scheme

        _update_diag_values_KKT!(KKT,map.diag_full,diag_kkt)

    end

    return is_success
end


function kktsolver_setrhs!(
    kktsolver::GPULDLKKTSolver{T},
    rhsx::CuVector{T},
    rhsz::CuVector{T}
) where {T}

    b = kktsolver.b
    (m,n,p) = (kktsolver.m,kktsolver.n,kktsolver.p)

    b[1:n]             .= rhsx
    b[(n+1):(n+m)]     .= rhsz
    b[(n+m+1):(n+m+p)] .= 0
    
    synchronize()

    return nothing
end


function kktsolver_getlhs!(
    kktsolver::GPULDLKKTSolver{T},
    lhsx::Union{Nothing,CuVector{T}},
    lhsz::Union{Nothing,CuVector{T}}
) where {T}

    x = kktsolver.x
    (m,n) = (kktsolver.m,kktsolver.n)

    isnothing(lhsx) || (@views lhsx .= x[1:n])
    isnothing(lhsz) || (@views lhsz .= x[(n+1):(n+m)])

    synchronize()

    return nothing
end


function kktsolver_solve!(
    kktsolver::GPULDLKKTSolver{T},
    lhsx::Union{Nothing,CuVector{T}},
    lhsz::Union{Nothing,CuVector{T}}
) where {T}

    (x,b) = (kktsolver.x,kktsolver.b)

    solve!(kktsolver.GPUsolver,x,b)

    is_success = begin
        if(kktsolver.settings.iterative_refinement_enable)
            #IR reports success based on finite normed residual
            is_success = _iterative_refinement(kktsolver,kktsolver.GPUsolver)
        else
             # otherwise must directly verify finite values
            is_success = all(isfinite,x)
        end
    end

    if is_success
       kktsolver_getlhs!(kktsolver,lhsx,lhsz)
    end

    return is_success
end

# update methods for P and A 
function kktsolver_update_P!(
    kktsolver::GPULDLKKTSolver{T},
    P::CuSparseMatrix{T}
) where{T}
    _update_values!(kktsolver.GPUsolver,kktsolver.KKT,kktsolver.map.P,P.nzVal)
end

function kktsolver_update_A!(
    kktsolver::GPULDLKKTSolver{T},
    A::CuSparseMatrix{T}
) where{T}
    _update_values!(kktsolver.GPUsolver,kktsolver.KKT,kktsolver.map.A,A.nzVal)
end

function kktsolver_update_At!(
    kktsolver::GPULDLKKTSolver{T},
    At::CuSparseMatrix{T}
) where{T}
    _update_values!(kktsolver.GPUsolver,kktsolver.KKT,kktsolver.map.At,At.nzVal)
end

function  _iterative_refinement(
    kktsolver::GPULDLKKTSolver{T},
    GPUsolver::AbstractDirectLDLSolver{T}
) where{T}

    (x,b)   = (kktsolver.x,kktsolver.b)
    (e,dx)  = (kktsolver.work1, kktsolver.work2)
    settings = kktsolver.settings

    #iterative refinement params
    IR_reltol    = settings.iterative_refinement_reltol
    IR_abstol    = settings.iterative_refinement_abstol
    IR_maxiter   = settings.iterative_refinement_max_iter
    IR_stopratio = settings.iterative_refinement_stop_ratio

    KKT = kktsolver.KKT
    normb  = norm(b,Inf)

    #compute the initial error
    norme = _get_refine_error!(e,b,KKT,x)
    # norme = _get_refine_error!(kktsolver,e,b,x) 
    isfinite(norme) || return is_success = false

    for i = 1:IR_maxiter

        if(norme <= IR_abstol + IR_reltol*normb)
            # within tolerance, or failed.  Exit
            break
        end
        lastnorme = norme

        #make a refinement and continue
        solve!(GPUsolver,dx,e)

        #prospective solution is x + dx.   Use dx space to
        #hold it for a check before applying to x
        @. dx += x
        synchronize()
        norme = _get_refine_error!(e,b,KKT,dx)
        # norme = _get_refine_error!(kktsolver,e,b,dx) 
        isfinite(norme) || return is_success = false

        improved_ratio = lastnorme/norme
        if(improved_ratio <  IR_stopratio)
            #insufficient improvement.  Exit
            if (improved_ratio > one(T))
                (x,dx) = (dx,x)   #pointer swap
            end
            break
        end
        (x,dx) = (dx,x)           #pointer swap
    end

    # make sure kktsolver fields now point to the right place
    # following possible swaps.   This is not necessary in the
    # Rust implementation since implementation there is via borrow
    (kktsolver.x,kktsolver.work2) = (x,dx)
 
    #NB: "success" means only that we had a finite valued result
    return is_success = true
end


# # computes e = b - Kξ, overwriting the first argument
# # and returning its norm

function _get_refine_error!(
    e::CuVector{T},
    b::CuVector{T},
    KKT::CuSparseMatrix{T},
    ξ::CuVector{T}) where {T}

    
    mul!(e,KKT,ξ)    # e = b - Kξ
    @. e = b - e
    synchronize()
    norme = norm(e,Inf)

    return norme
end

function _get_refine_error!(
    kktsolver::GPULDLKKTSolver{T},
    e::CuVector{T},
    b::CuVector{T},
    ξ::CuVector{T}) where {T}

    m = kktsolver.m 
    n = kktsolver.n
    p = kktsolver.p

    map = kktsolver.map
    KKT = kktsolver.KKT
    P = kktsolver.P
    A = kktsolver.A
    At = kktsolver.At

    cones = kktsolver.cones
    rng_cones = cones.rng_cones
    Matvut = cones.Matvut
    n_linear = cones.n_linear
    n_sparse_soc = cones.n_sparse_soc

    e1 = view(e, 1:n)
    e2 = view(e, (n+1):(n+m))
    ξ1 = view(ξ, 1:n)
    ξ2 = view(ξ, (n+1):(n+m))
    
    # e = b - KKT*ξ
    # mul!(e,KKT,ξ)  is split into several matrix-vector operations  
    mul!(e1, At, ξ2)
    CUDA.@sync @. e2 = 0
    Clarabel.mul_Hs_diag!(cones, e2, ξ2)
    mul!(e1, P, ξ1, one(T), one(T))
    mul!(e2, A, ξ1, one(T), -one(T))

    n_sparse_soc = cones.n_sparse_soc

    CUDA.@allowscalar if n_sparse_soc > 0
        rng_ext = (n+m+1):(n+m+p)
        e3 = view(e, rng_ext)
        ξ3 = view(ξ, rng_ext)

        rng_sparse_cone = (rng_cones[n_linear+1].start:rng_cones[n_linear+n_sparse_soc].stop) .+ n
        #Corresponding diagonal part for sparse SOCs 
        @views @. e[rng_sparse_cone] += KKT.nzVal[map.diag_full[rng_sparse_cone]]*ξ[rng_sparse_cone]
        @views @. e3 = KKT.nzVal[map.diag_full[rng_ext]]*ξ3
        synchronize()

        #Extended vu parts
        mul!(e2, Matvut', ξ3, one(T), one(T))
        mul!(e3, Matvut, ξ2, one(T), one(T))
    end

    CUDA.@sync @. e = b - e
    norme = norm(e,Inf)

    return norme
end

# update offdiagonal terms for socs in KKT
function _update_KKT_sparse_expandable(
    GPUsolver::AbstractDirectLDLSolver{T},
    KKT::CuSparseMatrix{T},
    map::GPUDataMap,
    cones::CompositeConeGPU{T}
) where {T}
    n_sparse_soc = cones.n_sparse_soc
    n_shift = cones.n_linear
    if n_sparse_soc > SPARSE_SOC_PARALELL_NUM
        update_KKT_sparse_soc_parallel!(KKT.nzVal, cones.η, map.D, map.vu, map.vut, cones.vut, cones.rng_cones, cones.numel_linear, n_shift, n_sparse_soc)
    elseif n_sparse_soc > 0
        update_KKT_sparse_soc_parallel_medium(KKT.nzVal, cones.η, map.D, map.vu, map.vut, cones.vut, cones.rng_cones, n_shift, n_sparse_soc)
    end
end

#################################################################
# kernel functions for GPU parallel computation w.r.t. ldl
#################################################################
#Parallel update for KKT of the sparse socs
function _kernel_update_KKT_sparse_soc_parallel!(
    nzVal::AbstractVector{T},
    η::AbstractVector{T},
    D::AbstractVector{Cint},
    vu::AbstractVector{Cint},
    vut::AbstractVector{Cint},
    vutval::AbstractVector{T},
    rng_cones::AbstractVector,
    numel_linear::Cint,
    n_shift::Cint,
    n_sparse_soc::Cint
) where {T}

    i = (blockIdx().x-one(Cint))*blockDim().x+threadIdx().x

    if i <= n_sparse_soc
        η2 = η[i]*η[i]

        shift_i = i + n_shift
        rng_i = rng_cones[shift_i] .- numel_linear
        endidx = 2*rng_i.stop
        startidx = endidx - 2*length(rng_i) + 1

        @inbounds for j in startidx:endidx
            val = vutval[j]
            nzVal[vu[j]] = val
            nzVal[vut[j]] = val
        end
    
        #set diagonal to η^2*(-1,1) in the extended rows/cols
        nzVal[D[2*i-1]] = -η2
        nzVal[D[2*i]] = η2
    end

    return nothing
end

@inline function update_KKT_sparse_soc_parallel!(
    nzVal::AbstractVector{T},
    η::AbstractVector{T},
    D::AbstractVector{Cint},
    vu::AbstractVector{Cint},
    vut::AbstractVector{Cint},
    vutval::AbstractVector{T},
    rng_cones::AbstractVector,
    numel_linear::Cint,
    n_shift::Cint,
    n_sparse_soc::Cint
) where {T}
    
    kernel = @cuda launch=false _kernel_update_KKT_sparse_soc_parallel!(nzVal, η, D, vu, vut, vutval, rng_cones, numel_linear, n_shift, n_sparse_soc)
    config = launch_configuration(kernel.fun)
    threads = min(n_sparse_soc, config.threads)
    blocks = cld(n_sparse_soc, threads)

    CUDA.@sync kernel(nzVal, η, D, vu, vut, vutval, rng_cones, numel_linear, n_shift, n_sparse_soc; threads, blocks)

end

###########################################################
# update_KKT_sparse_soc (several large socs)
###########################################################
@inline function update_KKT_sparse_soc_parallel_medium(
    nzval::AbstractVector{T}, 
    η::AbstractVector{T}, 
    mapD::AbstractVector{Cint},
    mapvu::AbstractVector{Cint}, 
    mapvut::AbstractVector{Cint}, 
    vut::AbstractVector{T}, 
    rng_cones::AbstractVector,
    n_shift::Cint, 
    n_sparse_soc::Cint
) where {T}
    #initialize kernel
    dummy_int = Cint(1024)
    kernel = @cuda launch=false _kernel_parent_update_KKT_sparse_soc(nzval, η, mapD, mapvu, mapvut, vut, rng_cones, n_shift, n_sparse_soc, dummy_int)
    config = launch_configuration(kernel.fun)
    threads = (1)
    blocks = (n_sparse_soc)

    CUDA.@sync kernel(nzval, η, mapD, mapvu, mapvut, vut, rng_cones, n_shift, n_sparse_soc, Cint(config.threads); threads, blocks)

end

function _kernel_parent_update_KKT_sparse_soc(
    nzval::AbstractVector{T}, 
    η::AbstractVector{T}, 
    mapD::AbstractVector{Cint},
    mapvu::AbstractVector{Cint}, 
    mapvut::AbstractVector{Cint}, 
    vut::AbstractVector{T}, 
    rng_cones::AbstractVector,
    n_shift::Cint, 
    n_sparse_soc::Cint, 
    maxthread::Cint
) where {T}
    tidx = (blockIdx().x - one(Cint)) * blockDim().x + threadIdx().x
    if tidx <= n_sparse_soc
        shift = n_shift + tidx

        #size should be doubled for v,u
        offset_tid = (rng_cones[shift].start - rng_cones[n_shift + one(Cint)].start)*Cint(2)  
        len_tid = (rng_cones[shift].stop - rng_cones[shift].start + one(Cint))*Cint(2)

        #thread-block info for cone-wise operations
        thread = min(len_tid, maxthread)
        block_uniform = cld(len_tid, thread)

        # update D blocks
        η2 = η[tidx]*η[tidx]
        nzval[mapD[Cint(2)*tidx - one(Cint)]] = -η2
        nzval[mapD[Cint(2)*tidx]] = η2

        @cuda threads = thread blocks = (2*block_uniform) dynamic = true _kernel_child_update_KKT_sparse_soc(vut, nzval, mapvu, mapvut, offset_tid, len_tid, block_uniform)
    end
    
    return nothing    
end

function _kernel_child_update_KKT_sparse_soc(
    vut::AbstractVector{T}, 
    nzval::AbstractVector{T}, 
    mapvu::AbstractVector{Cint}, 
    mapvut::AbstractVector{Cint}, 
    offset::Cint, 
    len::Cint,
    block_shift::Cint
) where {T}
    # vu part
    if (blockIdx().x <= block_shift)
        tid = (blockIdx().x-one(Cint))*blockDim().x+threadIdx().x
        if (tid <= len)
            nzval[mapvu[tid+offset]] = vut[tid+offset]
        end
    # vut part
    else
        tid = (blockIdx().x - block_shift -one(Cint))*blockDim().x+threadIdx().x
        if (tid <= len)
            nzval[mapvut[tid+offset]] = vut[tid+offset]
        end
    end
    return nothing
end

###########################################################
# update_KKT_sparse_soc (Huge socs)
###########################################################

#Sequential update for KKT of the sparse socs
@inline function update_KKT_sparse_soc_sequential!(
    GPUsolver::AbstractDirectLDLSolver{T},
    KKT::CuSparseMatrix{T},
    map::GPUDataMap,
    cones::CompositeConeGPU{T}
) where {T}
    
    n_shift = cones.n_linear

    sparse_idx = 0
    prow = one(Cint)
    CUDA.@allowscalar for i in 1:cones.n_sparse_soc
        ηi = cones.η[i]
        η2 = ηi*ηi

        shift_i = i + n_shift
        len_i = length(cones.rng_cones[shift_i])
        rng_i = (sparse_idx+1):(sparse_idx+2*len_i)

        vu_idx = view(map.vu, rng_i)
        vut_idx = view(map.vut, rng_i)
        vut_val = view(cones.vut, rng_i)

        _update_values!(GPUsolver,KKT,vu_idx,vut_val)
        _update_values!(GPUsolver,KKT,vut_idx,vut_val)

        #set diagonal to η^2*(-1,1) in the extended rows/cols
        _update_value!(KKT,map.D[prow], -η2)
        _update_value!(KKT,map.D[prow+1], η2)

        sparse_idx += 2*len_i
        prow += 2
    end

    synchronize()
end