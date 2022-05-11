# -----------------------------------------------------
# dispatch operators for multiple cones
# -----------------------------------------------------

function cones_all_symmetric(cones::ConeSet{T}) where {T}
    return any(is_symmetric, cones)
end

function cones_rectify_equilibration!(
    cones::ConeSet{T},
     δ::ConicVector{T},
     e::ConicVector{T}
) where{T}

    any_changed = false

    #we will update e <- δ .* e using return values
    #from this function.  default is to do nothing at all
    δ .= 1

    for i = eachindex(cones)
        any_changed |= rectify_equilibration!(cones[i],δ.views[i],e.views[i])
    end

    return any_changed
end


function cones_update_scaling!(
    cones::ConeSet{T},
    s::ConicVector{T},
    z::ConicVector{T},
	μ::T
) where {T}

    # update cone scalings by passing subview to each of
    # the appropriate cone types.
    for i = eachindex(cones)
        update_scaling!(cones[i],s.views[i],z.views[i],μ)
    end

    return nothing
end


function cones_set_identity_scaling!(
    cones::ConeSet{T}
) where {T}

    for i = eachindex(cones)
        set_identity_scaling!(cones[i])
    end

    return nothing
end


# The WtW block for each cone.
function cones_get_WtW_blocks!(
    cones::ConeSet{T},
    WtWblocks::Vector{Vector{T}}
) where {T}

    for i = eachindex(cones)
        get_WtW_block!(cones[i],WtWblocks[i])
    end
    return nothing
end

# YC:x = λ ∘ λ for symmetric cone and x = s for unsymmetric cones
function cones_affine_ds!(
    cones::ConeSet{T},
    x::ConicVector{T},
    s::ConicVector{T}
) where {T}

    for i = eachindex(cones)
        affine_ds!(cones[i],x.views[i],s.views[i])
    end
    return nothing
end

# YC:   x = y ∘ z for symmetric cones
#       x = 3rd-correction for unsymmetric cones
# NB: could merge with 3rd-functions later
function cones_circ_op!(
    cones::ConeSet{T},
    x::ConicVector{T},
    y::ConicVector{T},
    z::ConicVector{T}
) where {T}

    for i = eachindex(cones)
        # don't implement it for unsymmetric cones
        if !(cones.types[i] in NonsymmetricCones)
            circ_op!(cones[i],x.views[i],y.views[i],z.views[i])
        end
    end
    return nothing
end

# x = λ \ z,  where λ is scaled internal
# variable for each cone
function cones_λ_inv_circ_op!(
    cones::ConeSet{T},
    x::ConicVector{T},
    z::ConicVector{T}
) where {T}

    for i = eachindex(cones)
        # don't implement it for unsymmetric cones
        if !(cones.types[i] in NonsymmetricCones)
            λ_inv_circ_op!(cones[i],x.views[i],z.views[i])
        end
    end
    return nothing
end

# x = y \ z
function cones_inv_circ_op!(
    cones::ConeSet{T},
    x::ConicVector{T},
    y::ConicVector{T},
    z::ConicVector{T}
) where {T}

    for i = eachindex(cones)
        # don't implement it for unsymmetric cones
        if !(cones.types[i] in NonsymmetricCones)
            inv_circ_op!(cones[i],x.views[i],y.views[i],z.views[i])
        end
    end
    return nothing
end

# place a vector to some nearby point in the cone
# YC: only when there is no unsymmetric cone
function cones_shift_to_cone!(
    cones::ConeSet{T},
    z::ConicVector{T}
) where {T}

    for i = eachindex(cones)
        shift_to_cone!(cones[i],z.views[i])
    end
    return nothing
end

# initialization when with unsymmetric cones
function unit_initialization!(
    cones::ConeSet{T},
    s::ConicVector{T},
    z::ConicVector{T}
) where {T}

    for i = eachindex(cones)
        unsymmetric_init!(cones[i],s.views[i],z.views[i])
    end
    return nothing
end

# computes y = αWx + βy, or y = αWᵀx + βy, i.e.
# similar to the BLAS gemv interface.
#Warning: x must not alias y.
function cones_gemv_W!(
    cones::ConeSet{T},
    is_transpose::Symbol,
    x::ConicVector{T},
    y::ConicVector{T},
    α::T,
    β::T
) where {T}

    #@assert (x !== y)
    for i = eachindex(cones)
        # don't implement it for unsymmetric cones
        if !(cones.types[i] in NonsymmetricCones)
            gemv_W!(cones[i],is_transpose,x.views[i],y.views[i],α,β)
        end
    end
    return nothing
end

# compute ds in the combined step where λ ∘ (WΔz + W^{-⊤}Δs) = - ds
function cones_combined_ds!(
    cones::ConeSet{T},
    dz::ConicVector{T},
    ds::ConicVector{T},
    step_z::ConicVector{T},
    step_s::ConicVector{T},
    σμ::T
) where {T}
    
    for i = eachindex(cones)
        #Indeed, we compute the centering and the higher order correction parts in ds and save it in dz
        combined_ds!(cones[i],dz.views[i],step_z.views[i],step_s.views[i],σμ) 
    end

    #We are relying on d.s = λ ◦ λ (symmetric) or d.s = s (unsymmetric) already from the affine step here
    ds .+= dz                                 

    return nothing
end

# compute the generalized step Wᵀ(λ \ ds)
function cones_Wt_λ_inv_circ_ds!(
    cones::ConeSet{T},
    lz::ConicVector{T},
    rz::ConicVector{T},
    rs::ConicVector{T},
    Wtlinvds::ConicVector
) where {T}

    for i = eachindex(cones)
        Wt_λ_inv_circ_ds!(cones[i],lz.views[i],rz.views[i],rs.views[i],Wtlinvds.views[i]) 
    end

    return nothing
end

# compute the generalized step of -WᵀWΔz
function cones_WtW_Δz!(
    cones::ConeSet{T},
    lz::ConicVector{T},
    ls::ConicVector{T},
    workz::ConicVector{T}
) where {T}

    for i = eachindex(cones)
        WtW_Δz!(cones[i],lz.views[i],ls.views[i],workz.views[i])
    end

    return nothing
end

# maximum allowed step length over all cones
function cones_step_length(
    cones::ConeSet{T},
    dz::ConicVector{T},
    ds::ConicVector{T},
    dτ::T,
    dκ::T,
     z::ConicVector{T},
     s::ConicVector{T},
     τ::T,
     κ::T,
    α::T
) where {T}
    dz    = dz.views
    ds    = ds.views
    z     = z.views
    s     = s.views


    # YC: implement step search for symmetric cones first
    # NB: split the step search for symmetric and unsymmtric cones due to the complexity of the latter
    for i = eachindex(cones)
        if (cones.types[i] in NonsymmetricCones)
            αzs = unsymmetric_step_length(cones[i],dz[i],ds[i],z[i],s[i],α,cones.scaling)
            α = min(α,αzs)
        else
            (nextαz,nextαs) = step_length(cones[i],dz[i],ds[i],z[i],s[i])
            α = min(α,nextαz,nextαs)
        end
    end

    return α
end

# check the distance to the boundary for unsymmetric cones
function check_μ_and_centrality(
    cones::ConeSet{T},
    step::DefaultVariables{T},
    variables::DefaultVariables{T},
    work::DefaultVariables{T},
    α::T,
    steptype::Symbol
) where {T}

    dz    = step.z
    ds    = step.s
    dτ    = step.τ
    dκ    = step.κ
    z     = variables.z
    s     = variables.s
    τ     = variables.τ
    κ     = variables.κ
    cur_z = work.z
    cur_s = work.s

    zs= dot(z,s)
    dzs = dot(dz,ds)
    s_dz = dot(s,dz)
    z_ds = dot(z,ds)

    central_coef = cones.degree + 1

    # YC: scaling parameter to avoid reaching the boundary of cones
        # when we compute barrier functions
    # NB: different choice of α yields different performance, don't know how to explain it,
    #       but we must need it. Otherwise, there would be numerical issues for barrier computation
    α *= T(0.995)

    length_exp = cones.type_counts[ExponentialConeT]
    ind_exp = cones.ind_exp
    length_pow = cones.type_counts[PowerConeT]
    ind_pow = cones.ind_pow
    scaling = cones.scaling
    η = cones.η
    
    for j = 1:50
        #Initialize μ
        μ = (zs + τ*κ + α*(s_dz + z_ds + dτ*κ + τ*dκ) + α^2*(dzs + dτ*dκ))/central_coef
        upper = cones.minDist*μ     #bound for boundary distance

        @. cur_z = z + α*dz
        @. cur_s = s + α*ds

        # #boundary check from ECOS and centrality check from Hypatia
        # # NB:   1) the update x+α*dx is inefficient right now and need to be rewritten later
        # #       2) symmetric cones use the central path as in CVXOPT
        # if boundary_check!(cur_z,cur_s,ind_exp,length_exp,upper) && boundary_check!(cur_z,cur_s,ind_pow,length_pow,upper) && check_centrality!(cones,cur_s,cur_z,μ,η)
        #     return α
        # else
        #     α *= scaling
        # end
        

        # ECOS: check centrality, functional proximity measure
        # NB: the update x+α*dx is inefficient right now and need to be rewritten later
        if !(boundary_check!(cur_z,cur_s,ind_exp,length_exp,upper) && boundary_check!(cur_z,cur_s,ind_pow,length_pow,upper))
            α *= scaling
            continue
        end
        barrier = central_coef*log(μ) - log(τ+α*dτ) - log(κ+α*dκ)
        for i = eachindex(cones)
            barrier += f_sum(cones[i], cur_s.views[i], cur_z.views[i])
        end

        if barrier < 1.
            return α
        else
            α *= scaling    #backtrack line search
        end
        # println("centrality quite bad: ", barrier, " with ", central_coef)

    end

    if (steptype == :combined)
        error("get stalled with step size ", α)
    end

    return α
end

function boundary_check!(z,s,ind_cone,length_cone,upper)     
    for i = 1:length_cone
        μi = dot(z.views[ind_cone[i]],s.views[ind_cone[i]])/3

        # ECOS: if too close to boundary
        if μi < upper
            println("var too close to boundary")
            return false
        end
    end
    
    return true
end

function check_centrality!(cones,s,z,μ,η)
    for i in eachindex(cones)
        if !_check_neighbourhood(cones[i],s.views[i],z.views[i],μ,η)
            # println("centrality violation at cone ",i, "  ",cones.types[i])
            return false
        end
    end

    return true
end