using CUDA, CUDA.CUSPARSE
using CUDSS

export CUDSSDirectLDLSolver
struct CUDSSDirectLDLSolver{T} <: AbstractDirectLDLSolver{T}

    KKT::CuSparseMatrix{T}
    cudssSolver::CUDSS.CudssSolver{T}
    x::CuVector{T}
    b::CuVector{T}
    

    function CUDSSDirectLDLSolver{T}(KKT::CuSparseMatrix{T},x,b) where {T}

        LinearAlgebra.checksquare(KKT)

        #make a logical factorization to fix memory allocations
        # "S" denotes real symmetric and 'U' denotes the upper triangular

        cudssSolver = CUDSS.CudssSolver(KKT, "S", 'F')
        cudss_set(cudssSolver, "hybrid_execute_mode", true)    # only for CUDSS.jl V0.5.0
        cudss("analysis", cudssSolver, x, b)
        cudss("factorization", cudssSolver, x, b)

        return new(KKT,cudssSolver,x,b)
    end

end

ldlsolver_constructor(::Val{:cudss}) = CUDSSDirectLDLSolver
ldlsolver_matrix_shape(::Val{:cudss}) = :full

#refactor the linear system
function refactor!(ldlsolver::CUDSSDirectLDLSolver{T}) where{T}

    # Update the KKT matrix in the cudss solver
    cudss_set(ldlsolver.cudssSolver.matrix,ldlsolver.KKT)

    # Refactorization
    cudss("refactorization", ldlsolver.cudssSolver, ldlsolver.x, ldlsolver.b)

    # YC: should be corrected later on 
    return true
    # return all(isfinite, cudss_get(ldlsolver.cudssSolver.data,"diag"))

end


#solve the linear system
function solve!(
    ldlsolver::CUDSSDirectLDLSolver{T},
    x::CuVector{T},
    b::CuVector{T}
) where{T}
    
    #solve on GPU
    #ldiv!(x,ldlsolver.cudssSolver,b)
    cudss("solve", ldlsolver.cudssSolver, x, b)

end
