using Test, LinearAlgebra, Random, SparseArrays

using Clarabel

# using StatProfilerHTML
# include("../../src\\Clarabel.jl")
#only test this once
FloatT = Float64

function SOCP_lasso_data(Type::Type{T}) where {T <: AbstractFloat}

    #make all problem data Float64 and then convert.
    #otherwise random numbers won't be the same and different
    #instances will have different optimal values

    # generate problem data
    rng = Random.MersenneTwister(12345)
    n = 2
    m = 50 * n
    F = rand(rng, T, m, n)

    vtrue = sprand(rng, T, n, 1, 0.1)
    noise = T(0.1) * rand(rng, T, m, 1)
    b = F * vtrue + noise
    μMax = norm(F' * b, Inf)
    μ = T(0.1) * μMax

    # define lasso problem as SOCP
    A1 = -sparse([1 zeros(T, 1, 2 * n + 1) 1 zeros(T, 1, m);
    -1 zeros(T, 1, 2 * n) 1 zeros(T, 1, m + 1);
    zeros(T, m, 1) -2 * F zeros(T, m, n + 2) diagm( 0 => ones(T, m))])

    A2 = -sparse([zeros(T, n, 1) diagm(0 => ones(T, n)) -diagm(0 => ones(T, n)) zeros(T, n, m + 2);
    zeros(T, n, 1) -diagm(0 => ones(T, n)) -diagm(0 => ones(T, n)) zeros(T, n, m + 2)])
    A3 = -sparse([zeros(T, 1, 2 * n + 1) -one(T) zeros(T, 1, m + 1);
    zeros(T, 1, 2 * n + 2) -one(T) zeros(T, 1, m);
    zeros(T, m, 2 * n + 3) -diagm( 0 => ones(T, m))])
    b1 = T[1; 1; -2 * b[:]]
    b2 = zeros(T, 2 * n)
    b3 = zeros(T, m + 2)

    c = [one(T); zeros(T, n); μ * ones(T, n); zeros(T, m + 2)]
    P = spzeros(T, length(c), length(c))
    P = sparse(I(length(c))*one(T)*1e-6)

    A = [A1;A2;A3]
    b = [b1;b2;b3]

    cones = [Clarabel.NonnegativeConeT(length(b1)),
             Clarabel.NonnegativeConeT(length(b2)),
             Clarabel.SecondOrderConeT(length(b3))]



    return (P,c,A,b,cones)

end

settings = Clarabel.Settings(
        equilibrate_enable=false,
        direct_kkt_solver=true,
        static_regularization_enable=false,
        presolve_enable=false,
        # iterative_refinement_enable = false,
        tol_gap_abs = 1e-5,
        tol_gap_rel = 1e-5,
        tol_feas = 1e-5,
        tol_ktratio = 1e-4
        )
P,c,A,b,cones = SOCP_lasso_data(FloatT)
solver   = Clarabel.Solver(P,c,A,b,cones,settings)
# solver.settings.direct_kkt_solver = false
Clarabel.solve!(solver)

# solver.solution.status == Clarabel.SOLVED
