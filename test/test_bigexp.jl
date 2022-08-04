using ECOS, Mosek, MosekTools
using JuMP, MathOptInterface
# const MOI = MathOptInterface
using LinearAlgebra
using ConicBenchmarkUtilities

using Profile,StatProfilerHTML, TimerOutputs

#include("../src\\Clarabel.jl")
using Clarabel
# using Hypatia

coneMap = Dict(:Zero => MOI.Zeros, :Free => :Free,
                     :NonPos => MOI.Nonpositives, :NonNeg => MOI.Nonnegatives,
                     :SOC => MOI.SecondOrderCone, :SOCRotated => MOI.RotatedSecondOrderCone,
                     :ExpPrimal => MOI.ExponentialCone, :ExpDual => MOI.DualExponentialCone)

function exp_model(exInd::Int)
    filelist = readdir(pwd()*"./primal_exp_cbf")

    datadir = filelist[exInd]   #"gp_dave_1.cbf.gz"
    dat = readcbfdata("./primal_exp_cbf/"*datadir) # .cbf.gz extension also accepted

    # In MathProgBase format:
    c, A, b, con_cones, var_cones, vartypes, sense, objoffset = cbftompb(dat)
    # Note: The sense in MathProgBase form is always minimization, and the objective offset is zero.
    # If sense == :Max, you should flip the sign of c before handing off to a solver.
    if sense == :Max
        c .*= -1
    end

    num_con = size(A,1)
    num_var = size(A,2)

    model = Model(Clarabel.Optimizer)
    set_optimizer_attribute(model, "direct_solve_method", :qdldl)

    # model = Model(ECOS.Optimizer)
    @variable(model, x[1:num_var])

    #Tackling constraint
    for i = 1:length(con_cones)
        cur_cone = con_cones[i]
        # println(coneMap[cur_cone[1]])

        if coneMap[cur_cone[1]] == :Free
            continue
        elseif coneMap[cur_cone[1]] == MOI.ExponentialCone
            @constraint(model, b[cur_cone[2]] - A[cur_cone[2],:]*x in MOI.ExponentialCone())
        # elseif coneMap[cur_cone[1]] == MOI.DualExponentialCone
        #     @constraint(model, b[cur_cone[2]] - A[cur_cone[2],:]*x in MOI.DualExponentialCone())
        else
            @constraint(model, b[cur_cone[2]] - A[cur_cone[2],:]*x in coneMap[cur_cone[1]](length(cur_cone[2])))
        end
    end

    for i = 1:length(var_cones)
        cur_var = var_cones[i]
        # println(coneMap[cur_var[1]])

        if coneMap[cur_var[1]] == :Free
            continue
        elseif coneMap[cur_var[1]] == MOI.ExponentialCone
            @constraint(model, x[cur_var[2]] in MOI.ExponentialCone())
        # elseif coneMap[cur_var[1]] == MOI.DualExponentialCone
        #     @constraint(model, x[cur_var[2]] in MOI.DualExponentialCone())
        else
            @constraint(model, x[cur_var[2]] in coneMap[cur_var[1]](length(cur_var[2])))
        end
    end

    @objective(model, Min, sum(c.*x))

    return model
end

# the input number i corresponds to the i-th example in CBLIB. Example 7,8,32
model = exp_model(7) 
Profile.clear()
Profile.init()
@profilehtml optimize!(model)
