
function print_banner(io::IO, verbose::Bool)

    if !verbose return; end
    println(io, "-------------------------------------------------------------")
    @printf(io, "           Clarabel.jl v%s  -  Clever Acronym              \n", version())
    println(io, "                   (c) Paul Goulart                          ")
    println(io, "                University of Oxford, 2022                   ")
    println(io, "-------------------------------------------------------------")
end
print_banner(verbose) = print_banner(stdout,verbose)


function info_print_configuration(
    io::IO,
    info::DefaultInfo{T},
    settings::Settings{T},
    data::DefaultProblemData{T},
    cones::CompositeCone{T}
) where {T}

    if(settings.verbose == false) return end

    if(!isnothing(data.presolver))
        @printf(io, "\npresolve: removed %i constraints\n", count_reduced(data.presolver))
    end 

    if(!isnothing(data.chordal_info))
        print_chordal_decomposition(io, data.chordal_info, settings)
    end

    @printf(io, "\nproblem:\n")
    @printf(io, "  variables     = %i\n", data.n)
    @printf(io, "  constraints   = %i\n", data.m)
    @printf(io, "  nnz(P)        = %i\n", nnz(data.P))
    @printf(io, "  nnz(A)        = %i\n", nnz(data.A))
    @printf(io, "  cones (total) = %i\n", length(cones))
    print_conedims_by_type(io, cones, ZeroCone)
    print_conedims_by_type(io, cones, NonnegativeCone)
    print_conedims_by_type(io, cones, SecondOrderCone)
    print_conedims_by_type(io, cones, ExponentialCone)
    print_conedims_by_type(io, cones, PowerCone)
    print_conedims_by_type(io, cones, GenPowerCone)
    print_conedims_by_type(io, cones, PSDTriangleCone)
    print_settings(io, settings)
    @printf(io, "\n")

    return nothing
end

function info_print_configuration(
    io::IO,
    info::DefaultInfo{T},
    settings::Settings{T},
    data::DefaultProblemDataGPU{T},
    cones::CompositeConeGPU{T}
) where {T}

    if(settings.verbose == false) return end

    if(!isnothing(data.presolver))
        @printf(io, "\npresolve: removed %i constraints\n", count_reduced(data.presolver))
    end 

    # if(!isnothing(data.chordal_info))
    #     print_chordal_decomposition(io, data.chordal_info, settings)
    # end

    @printf(io, "\nproblem:\n")
    @printf(io, "  variables     = %i\n", data.n)
    @printf(io, "  constraints   = %i\n", data.m)
    @printf(io, "  nnz(P)        = %i\n", nnz(data.P))
    @printf(io, "  nnz(A)        = %i\n", nnz(data.A))
    @printf(io, "  cones (total) = %i\n", length(data.cones))
    print_conedims_by_type(io, cones, data.cones, Clarabel.ZeroConeT)
    print_conedims_by_type(io, cones, data.cones, Clarabel.NonnegativeConeT)
    print_conedims_by_type(io, cones, data.cones, Clarabel.SecondOrderConeT)
    print_conedims_by_type(io, cones, data.cones, Clarabel.ExponentialConeT)
    print_conedims_by_type(io, cones, data.cones, Clarabel.PowerConeT)
    print_conedims_by_type(io, cones, data.cones, Clarabel.GenPowerConeT)
    print_conedims_by_type(io, cones, data.cones, Clarabel.PSDTriangleConeT)
    print_settings(io, settings)
    @printf(io, "\n")

    return nothing
end

info_print_configuration(info,settings,data,cones) = info_print_configuration(stdout,info,settings,data,cones)


function info_print_status_header(
    io::IO,
    info::DefaultInfo{T},
    settings::Settings{T},
) where {T}

    if(settings.verbose == false) return end

    #print a subheader for the iterations info
    @printf(io, "%s", "iter    ")
    @printf(io, "%s", "pcost        ")
    @printf(io, "%s", "dcost       ")
    @printf(io, "%s", "gap       ")
    @printf(io, "%s", "pres      ")
    @printf(io, "%s", "dres      ")
    @printf(io, "%s", "k/t       ")
    @printf(io, "%s", " μ       ")
    @printf(io, "%s", "step      ")
    @printf(io, "\n")
    println(io, "---------------------------------------------------------------------------------------------")

    return nothing
end
info_print_status_header(info, settings) = info_print_status_header(stdout,info,settings)

function info_print_status(
    io::IO,
    info::DefaultInfo{T},
    settings::Settings
) where {T}

    if(settings.verbose == false) return end

    @printf(io, "%3d  ", info.iterations)
    @printf(io, "% .4e  ", info.cost_primal)
    @printf(io, "% .4e  ", info.cost_dual)
    @printf(io, "%.2e  ", min(info.gap_abs,info.gap_rel))
    @printf(io, "%.2e  ", info.res_primal)
    @printf(io, "%.2e  ", info.res_dual)
    @printf(io, "%.2e  ", info.ktratio)
    @printf(io, "%.2e  ", info.μ)
    if(info.iterations > 0)
        @printf(io, "%.2e  ", info.step_length)
    else
        @printf(io, " ------   ") #info.step_length
    end

    @printf(io, "\n")

    return nothing
end
info_print_status(info,settings) = info_print_status(stdout,info,settings)


function info_print_footer(
    io::IO,
    info::DefaultInfo{T},
    settings::Settings
) where {T}

    if(settings.verbose == false) return end

    println(io, "---------------------------------------------------------------------------------------------")
    @printf(io, "Terminated with status = %s\n",SolverStatusDict[info.status])
    @printf(io, "total time = %s",TimerOutputs.prettytime(info.total_time*1e9))
    @printf(io, " (setup time = %s,",TimerOutputs.prettytime(info.setup_time*1e9))
    @printf(io, " solve time = %s)\n",TimerOutputs.prettytime(info.solve_time*1e9))

    return nothing
end
info_print_footer(info,settings) = info_print_footer(stdout,info,settings)


function bool_on_off(v::Bool)
    return  v ? "on" : "off"
end


function print_settings(io::IO, settings::Settings{T}) where {T}

    set = settings
    @printf(io, "\nsettings:\n")

    @printf(io, "  linear algebra: direct / %s, precision: %s\n", set.direct_solve_method, get_precision_string(T))

    @printf(io, "  max iter = %i, time limit = %f,  max step = %.3f\n",
        set.max_iter,
        set.time_limit,
        set.max_step_fraction,
    )
    #
    @printf(io, "  tol_feas = %0.1e, tol_gap_abs = %0.1e, tol_gap_rel = %0.1e,\n",
        set.tol_feas,
        set.tol_gap_abs,
        set.tol_gap_rel
    )

    @printf(io, "  static reg : %s, ϵ1 = %0.1e, ϵ2 = %0.1e\n",
        bool_on_off(set.static_regularization_enable),
        set.static_regularization_constant,
        set.static_regularization_proportional,

    )
    #
    @printf(io, "  dynamic reg: %s, ϵ = %0.1e, δ = %0.1e\n",
        bool_on_off(set.dynamic_regularization_enable),
        set.dynamic_regularization_eps,
        set.dynamic_regularization_delta
    )
    @printf(io, "  iter refine: %s, reltol = %0.1e, abstol = %0.1e, \n",
        bool_on_off(set.iterative_refinement_enable),
        set.iterative_refinement_reltol,
        set.iterative_refinement_abstol
    )
    @printf(io, "               max iter = %d, stop ratio = %.1f\n",
        set.iterative_refinement_max_iter,
        set.iterative_refinement_stop_ratio
    )
    @printf(io, "  equilibrate: %s, min_scale = %0.1e, max_scale = %0.1e\n",
        bool_on_off(set.equilibrate_enable),
        set.equilibrate_min_scaling,
        set.equilibrate_max_scaling
    )
    @printf(io, "               max iter = %d\n",
        set.equilibrate_max_iter,
    )

    return nothing
end

function print_chordal_decomposition(io::IO, chordal_info::ChordalInfo{T}, settings::Settings{T}) where {T}

    @printf(io, "\nchordal decomposition:\n")
    @printf(io, "  compact format = %s, dual completion = %s\n", 
            bool_on_off(settings.chordal_decomposition_compact), 
            bool_on_off(settings.chordal_decomposition_complete_dual))

    @printf(io, "  merge method = %s\n", settings.chordal_decomposition_merge_method) 

    @printf(io, "  PSD cones initial             = %d\n",
            init_psd_cone_count(chordal_info))
    @printf(io, "  PSD cones decomposable        = %d\n",
            decomposable_cone_count(chordal_info))
    @printf(io, "  PSD cones after decomposition = %d\n",
            premerge_psd_cone_count(chordal_info))
    @printf(io, "  PSD cones after merges        = %d\n",
            final_psd_cone_count(chordal_info))

end 


get_precision_string(T::Type{<:Real}) = string(T)
get_precision_string(T::Type{<:BigFloat}) = string(T," (", precision(T), " bit)")


function print_conedims_by_type(io::IO, cones::CompositeCone{T}, type::Type) where {T}

    maxlistlen = 5

    #how many of this type of cone?
    count = get_type_count(cones,type)

    #skip if there are none of this type
    if count == 0
        return #don't report if none
    end

    nvars = DefaultInt[Clarabel.numel(K) for K in cones[isa.(cones,type)]]
    name  = rpad(string(nameof(type))[1:end-4],11)  #drops "Cone" part
    @printf(io, "    : %s = %i, ", name, count)

    if count == 1
        @printf(io, " numel = %i",nvars[1])

    elseif count <= maxlistlen
        #print them all
        @printf(io, " numel = (")
        foreach(x->@printf(io, "%i,",x),nvars[1:end-1])
        @printf(io, "%i)",nvars[end])

    else
        #print first (maxlistlen-1) and the final one
        @printf(io, " numel = (")
        foreach(x->@printf(io, "%i,",x),nvars[1:(maxlistlen-1)])
        @printf(io, "...,%i)",nvars[end])
    end

    @printf(io, "\n")

end

function print_conedims_by_type(io::IO, cones::CompositeConeGPU{T}, cone_specs::Vector{SupportedCone}, type::DataType) where {T}

    maxlistlen = 5

    #how many of this type of cone?
    count = get_type_count(cones,type)
    #skip if there are none of this type
    if count == 0
        return #don't report if none
    end

    nvars = DefaultInt[Clarabel.nvars(K) for K in cone_specs[isa.(cone_specs,type)]]
    name  = rpad(string(nameof(ConeDict[type]))[1:end-4],11)  #drops "Cone" part
    @printf(io, "    : %s = %i, ", name, count)

    if count == 1
        @printf(io, " numel = %i",nvars[1])

    elseif count <= maxlistlen
        #print them all
        @printf(io, " numel = (")
        foreach(x->@printf(io, "%i,",x),nvars[1:end-1])
        @printf(io, "%i)",nvars[end])

    else
        #print first (maxlistlen-1) and the final one
        @printf(io, " numel = (")
        foreach(x->@printf(io, "%i,",x),nvars[1:(maxlistlen-1)])
        @printf(io, "...,%i)",nvars[end])
    end

    @printf(io, "\n")

end