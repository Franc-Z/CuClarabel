# ---------------
# KKT System
# ---------------

mutable struct DefaultKKTSystemGPU{T} <: AbstractKKTSystem{T}

    #the KKT system solver
    kktsolver::AbstractKKTSolver{T}

    #solution vector for constant part of KKT solves
    x1::CuVector{T}
    z1::CuVector{T}

    #solution vector for general KKT solves
    x2::CuVector{T}
    z2::CuVector{T}
    
    #work vectors for assembling/disassembling vectors
    workx::CuVector{T}
    workz::CuVector{T}
    work_conic::CuVector{T}

    #temporary GPU vector for the conic part 
    workx2::CuVector{T}

        function DefaultKKTSystemGPU{T}(
            data::DefaultProblemDataGPU{T},
            cones::CompositeConeGPU{T},
            settings::Settings{T}
        ) where {T}

        #basic problem dimensions
        (m, n) = (Cint(data.m), Cint(data.n))

        kktsolver = GPULDLKKTSolver{T}(data.P, data.A, data.At, cones, m, n, settings)

        #the LHS constant part of the reduced solve
        x1   = CuVector{T}(undef,n)
        z1   = CuVector{T}(undef,m)

        #the LHS for other solves
        x2   = CuVector{T}(undef,n)
        z2   = CuVector{T}(undef,m)

        #workspace compatible with (x,z)
        workx   = CuVector{T}(undef,n)
        workz   = CuVector{T}(undef,m)

        #additional conic workspace vector compatible with s and z
        work_conic = CuVector{T}(undef,m)

        workx2 = CuVector{T}(undef,n)

        return new(kktsolver,x1,z1,x2,z2,workx,workz,work_conic,workx2)

    end

end

DefaultKKTSystemGPU(args...) = DefaultKKTSystemGPU{DefaultFloat}(args...)

function kkt_linear_solver_info(kktsystem::DefaultKKTSystemGPU{T}) where{T}
    return kktsolver_linear_solver_info(kktsystem.kktsolver)
end

function kkt_update!(
    kktsystem::DefaultKKTSystemGPU{T},
    data::DefaultProblemDataGPU{T},
    cones::CompositeConeGPU{T}
) where {T}

    #update the linear solver with new cones
    is_success  = kktsolver_update!(kktsystem.kktsolver,cones)

    #bail if the factorization has failed 
    is_success || return is_success

    #calculate KKT solution for constant terms
    is_success = _kkt_solve_constant_rhs!(kktsystem,data)

    return is_success
end

function _kkt_solve_constant_rhs!(
    kktsystem::DefaultKKTSystemGPU{T},
    data::DefaultProblemDataGPU{T}
) where {T}

    CUDA.@sync @. kktsystem.workx = -data.q;

    kktsolver_setrhs!(kktsystem.kktsolver, kktsystem.workx, data.b)
    is_success = kktsolver_solve!(kktsystem.kktsolver, kktsystem.x2, kktsystem.z2)

    return is_success

end


function kkt_solve_initial_point!(
    kktsystem::DefaultKKTSystemGPU{T},
    variables::DefaultVariables{T},
    data::DefaultProblemDataGPU{T}
) where{T}

    if nnz(data.P) == 0
        # LP initialization
        # solve with [0;b] as a RHS to get (x,-s) initializers
        # zero out any sparse cone variables at end
        kktsystem.workx .= zero(T)
        kktsystem.workz .= data.b
        synchronize()
        
        kktsolver_setrhs!(kktsystem.kktsolver, kktsystem.workx, kktsystem.workz)
        is_success = kktsolver_solve!(kktsystem.kktsolver, variables.x, variables.s)
        variables.s .= -variables.s

        if !is_success return is_success end

        # solve with [-q;0] as a RHS to get z initializer
        # zero out any sparse cone variables at end
        @. kktsystem.workx = -data.q
        kktsystem.workz .=  zero(T)
        synchronize()

        kktsolver_setrhs!(kktsystem.kktsolver, kktsystem.workx, kktsystem.workz)
        is_success = kktsolver_solve!(kktsystem.kktsolver, nothing, variables.z)
    else
        # QP initialization
        @. kktsystem.workx = -data.q
        @. kktsystem.workz = data.b
        synchronize()

        kktsolver_setrhs!(kktsystem.kktsolver, kktsystem.workx, kktsystem.workz)
        is_success = kktsolver_solve!(kktsystem.kktsolver, variables.x, variables.z)
        @. variables.s = -variables.z
        synchronize()
    end

    return is_success

end

function kkt_solve!(
    kktsystem::DefaultKKTSystemGPU{T},
    lhs::DefaultVariables{T},
    rhs::DefaultVariables{T},
    data::DefaultProblemDataGPU{T},
    variables::DefaultVariables{T},
    cones::CompositeConeGPU{T},
    steptype::Symbol   #:affine or :combined
) where{T}

    (x1,z1) = (kktsystem.x1, kktsystem.z1)
    (x2,z2) = (kktsystem.x2, kktsystem.z2)
    (workx,workz) = (kktsystem.workx, kktsystem.workz)

    #solve for (x1,z1)
    #-----------
    @. workx = rhs.x

    # compute the vector c in the step equation HₛΔz + Δs = -c,  
    # with shortcut in affine case
    Δs_const_term = kktsystem.work_conic

    if steptype == :affine
        CUDA.@sync @. Δs_const_term = variables.s
    else  #:combined expected, but any general RHS should do this
        #we can use the overall LHS output as additional workspace for the moment
        Δs_from_Δz_offset!(cones,Δs_const_term,rhs.s,lhs.z,variables.z)
    end

    @. workz = Δs_const_term - rhs.z
    synchronize()


    #---------------------------------------------------
    #this solves the variable part of reduced KKT system
    kktsolver_setrhs!(kktsystem.kktsolver, workx, workz)
    is_success = kktsolver_solve!(kktsystem.kktsolver,x1,z1)

    if !is_success return false end

    #solve for Δτ.
    #-----------
    # Numerator first
    ξ   = workx
    @. ξ = variables.x / variables.τ
    synchronize()

    workx2 = kktsystem.workx2
    mul!(workx2,data.P,x1)
    tau_num = rhs.τ - rhs.κ/variables.τ + dot(data.q,x1) + dot(data.b,z1) + 2*dot(ξ,workx2)

    #offset ξ for the quadratic form in the denominator
    ξ_minus_x2    = ξ   #alias to ξ, same as workx
    @. ξ_minus_x2  -= x2
    synchronize()

    tau_den  = variables.κ/variables.τ - dot(data.q,x2) - dot(data.b,z2)
    mul!(workx2,data.P,ξ_minus_x2)
    t1 = dot(ξ_minus_x2,workx2)
    mul!(workx2,data.P,x2)
    t2 = dot(x2,workx2)
    tau_den += t1 - t2

    #solve for (Δx,Δz)
    #-----------
    lhs.τ  = tau_num/tau_den
    @. lhs.x = x1 + lhs.τ * x2
    @. lhs.z = z1 + lhs.τ * z2
    synchronize()

    #solve for Δs
    #-------------
    # compute the linear term HₛΔz, where Hs = WᵀW for symmetric
    # cones and Hs = μH(z) for asymmetric cones
    mul_Hs!(cones,lhs.s,lhs.z,workz)
    @. lhs.s = -(lhs.s + Δs_const_term)

    synchronize()
    
    #solve for Δκ
    #--------------
    lhs.κ = -(rhs.κ + variables.κ * lhs.τ) / variables.τ

    # we don't check the validity of anything
    # after the KKT solve, so just return is_success
    # without further validation
    return is_success

end

#update the KKT system with new P and A, At
function kkt_update_P!(
    kktsystem::DefaultKKTSystemGPU{T},
    P::CuSparseMatrix{T}
) where{T}
    kktsolver_update_P!(kktsystem.kktsolver,P)
    return nothing
end

function kkt_update_A!(
    kktsystem::DefaultKKTSystemGPU{T},
    A::CuSparseMatrix{T}
) where{T}
    kktsolver_update_A!(kktsystem.kktsolver,A)
    return nothing
end

function kkt_update_At!(
    kktsystem::DefaultKKTSystemGPU{T},
    At::CuSparseMatrix{T}
) where{T}
    kktsolver_update_At!(kktsystem.kktsolver,At)
    return nothing
end