# ## ------------------------------------
# # Nonnegative Cone
# # -------------------------------------

# degree(K::NonnegativeCone{T}) where {T} = K.dim
# numel(K::NonnegativeCone{T}) where {T} = K.dim

# function rectify_equilibration!(
#     K::NonnegativeCone{T},
#     δ::AbstractVector{T},
#     e::AbstractVector{T}
# ) where{T}

#     #allow elementwise equilibration scaling
#     δ .= one(T)
#     return false
# end

@inline function margins_nonnegative(
    z::AbstractVector{T},
    α::AbstractVector{T},
    rng_cones::AbstractVector,
    idx_inq::Vector{Cint},
    αmin::T
) where{T}
    margin = zero(T)
    
    CUDA.@allowscalar for i in idx_inq
        rng_cone_i = rng_cones[i]
        @views zi = z[rng_cone_i]
        # Use GPU reduction for minimum
        αmin = min(αmin, minimum(zi))
        @views αi = α[rng_cone_i]
        # Use GPU broadcast
        @. αi = max(zi, zero(T))
        # Use GPU reduction for sum
        margin += sum(αi)
    end

    return (αmin, margin)
end

# place vector into nn cone
@inline function scaled_unit_shift_nonnegative!(
    z::AbstractVector{T},
    rng_cones::AbstractVector,
    idx_inq::Vector{Cint},
    α::T
) where{T}

    CUDA.@allowscalar for i in idx_inq
        rng_cone_i = rng_cones[i]
        @views @. z[rng_cone_i] += α 
    end
end

# unit initialization for asymmetric solves
@inline function unit_initialization_nonnegative!(
   z::AbstractVector{T},
   s::AbstractVector{T},
   rng_cones::AbstractVector,
   idx_inq::Vector{Cint}
) where{T}

    CUDA.@allowscalar for i in idx_inq
        rng_cone_i = rng_cones[i]
        @views @. z[rng_cone_i] = one(T)
        @views @. s[rng_cone_i] = one(T)
    end
end

#configure cone internals to provide W = I scaling
@inline function set_identity_scaling_nonnegative!(
    w::AbstractVector{T},
    rng_cones::AbstractVector,
    idx_inq::Vector{Cint}
) where {T}

    CUDA.@allowscalar for i in idx_inq
        @views @. w[rng_cones[i]] = one(T)
    end
end

@inline function update_scaling_nonnegative!(
    s::AbstractVector{T},
    z::AbstractVector{T},
    w::AbstractVector{T},
    λ::AbstractVector{T},
    rng_cones::AbstractVector,
    idx_inq::Vector{Cint}    
) where {T}

    CUDA.@allowscalar for i in idx_inq
        rng_cone_i = rng_cones[i]
        @views si = s[rng_cone_i]
        @views zi = z[rng_cone_i]
        @views λi = λ[rng_cone_i]
        @views wi = w[rng_cone_i]
        # Use GPU broadcast operations
        @. λi = sqrt(si * zi)
        @. wi = sqrt(si / zi)
    end
end

@inline function get_Hs_nonnegative!(
    Hsblocks::AbstractVector{T},
    w::AbstractVector{T},
    rng_cones::AbstractVector,
    rng_blocks::AbstractVector,
    idx_inq::Vector{Cint}  
) where {T}

    #this block is diagonal, and we expect here
    #to receive only the diagonal elements to fill
    CUDA.@allowscalar for i in idx_inq
        @views wi = w[rng_cones[i]]
        @views @. Hsblocks[rng_blocks[i]] = wi^2
    end
end

# compute the product y = WᵀWx
@inline function mul_Hs_nonnegative!(
    y::AbstractVector{T},
    x::AbstractVector{T},
    w::AbstractVector{T},
    rng_cones::AbstractVector,
    idx_inq::Vector{Cint} 
) where {T}

    #NB : seemingly sensitive to order of multiplication
    CUDA.@allowscalar for i in idx_inq
        @views wi = w[rng_cones[i]]
        @views xi = x[rng_cones[i]]
        @views yi = y[rng_cones[i]]
        @. yi = wi * (wi * xi)
    end
end

# returns ds = λ∘λ for the nn cone
@inline function affine_ds_nonnegative!(
    ds::AbstractVector{T},
    λ::AbstractVector{T},
    rng_cones::AbstractVector,
    idx_inq::Vector{Cint} 
) where {T}

    CUDA.@allowscalar for i in idx_inq
        @views @. ds[rng_cones[i]] = λ[rng_cones[i]]^2
    end
end

@inline function combined_ds_shift_nonnegative!(
    shift::AbstractVector{T},
    step_z::AbstractVector{T},
    step_s::AbstractVector{T},
    w::AbstractVector{T},
    σμ::T,
    rng_cones::AbstractVector,
    idx_inq::Vector{Cint} 
) where {T}

    # The shift must be assembled carefully if we want to be economical with
    # allocated memory.  Will modify the step.z and step.s in place since
    # they are from the affine step and not needed anymore.

    CUDA.@allowscalar for i in idx_inq
        rng = rng_cones[i]
        @views shift_i = shift[rng]
        @views step_zi = step_z[rng]
        @views step_si = step_s[rng]
        @views wi = w[rng]

        # Use element-wise operations
        # Δz <- WΔz
        @. step_zi *= wi
        # Δs <- W⁻¹Δs
        @. step_si /= wi
        # shift = W⁻¹Δs ∘ WΔz - σμe
        @. shift_i = step_si * step_zi - σμ    
    end
end

@inline function Δs_from_Δz_offset_nonnegative!(
    out::AbstractVector{T},
    ds::AbstractVector{T},
    z::AbstractVector{T},
    rng_cones::AbstractVector,
    idx_inq::Vector{Cint} 
) where {T}
    CUDA.@allowscalar for i in idx_inq
        @views @. out[rng_cones[i]] = ds[rng_cones[i]] / z[rng_cones[i]]
    end
end

#return maximum allowable step length while remaining in the nn cone
function _kernel_step_length_nonnegative(
    dz::AbstractVector{T},
    ds::AbstractVector{T},
     z::AbstractVector{T},
     s::AbstractVector{T},
     α::AbstractVector{T},
     len_rng::Cint,
     αmax::T
) where {T}

    i = (blockIdx().x-1)*blockDim().x+threadIdx().x
    if i <= len_rng
        αz = dz[i] < 0 ? (min(αmax,-z[i]/dz[i])) : αmax
        αs = ds[i] < 0 ? (min(αmax,-s[i]/ds[i])) : αmax
        α[i] = min(αz, αs)
    end

    return nothing
end

@inline function step_length_nonnegative(
    dz::AbstractVector{T},
    ds::AbstractVector{T},
     z::AbstractVector{T},
     s::AbstractVector{T},
     α::AbstractVector{T},
     αmax::T,
     rng_cones::AbstractVector,
     idx_inq::Vector{Cint} 
) where {T}

    CUDA.@allowscalar for i in idx_inq
        len_nn = Cint(length(rng_cones[i]))
        rng_cone_i = rng_cones[i]
        @views dzi = dz[rng_cone_i]
        @views dsi = ds[rng_cone_i]
        @views zi = z[rng_cone_i]
        @views si = s[rng_cone_i]
        @views αi = α[rng_cone_i]
        
        # Only use kernel for large cones
        if len_nn > 256  # threshold for kernel launch
            kernel = @cuda launch=false _kernel_step_length_nonnegative(dzi, dsi, zi, si, αi, len_nn, αmax)
            config = launch_configuration(kernel.fun)
            threads = min(len_nn, config.threads)
            blocks = cld(len_nn, threads)
        
            CUDA.@sync kernel(dzi, dsi, zi, si, αi, len_nn, αmax; threads, blocks)
            αmax = min(αmax, minimum(αi))
        else
            # For small cones, use direct computation with GPU operations
            @. αi = ifelse(dzi < 0, min(αmax, -zi/dzi), αmax)
            @. αi = ifelse(dsi < 0, min(αi, -si/dsi), αi)
            αmax = min(αmax, minimum(αi))
        end
    end

    return αmax
end

function _kernel_compute_barrier_nonnegative(
    barrier::AbstractVector{T},
    z::AbstractVector{T},
    s::AbstractVector{T},
    dz::AbstractVector{T},
    ds::AbstractVector{T},
    α::T,
    len_nn::Cint
) where {T}

    i = (blockIdx().x-1)*blockDim().x+threadIdx().x
    if i <= len_nn
        sz_new = (s[i] + α*ds[i])*(z[i] + α*dz[i])
        barrier[i] = -logsafe_gpu(sz_new)
    end

    return nothing
end

@inline function compute_barrier_nonnegative(
    work::AbstractVector{T},
    z::AbstractVector{T},
    s::AbstractVector{T},
    dz::AbstractVector{T},
    ds::AbstractVector{T},
    α::T,
    rng_cones::AbstractVector,
    idx_inq::Vector{Cint},
    len_nn::Cint
) where {T}

    barrier = zero(T)
    CUDA.@allowscalar for i in idx_inq
        rng_cone_i = rng_cones[i]
        len_cone = length(rng_cone_i)
        
        if len_cone > 256  # threshold for kernel launch
            @views worki = work[rng_cone_i]
            @views zi = z[rng_cone_i]
            @views si = s[rng_cone_i]
            @views dzi = dz[rng_cone_i]
            @views dsi = ds[rng_cone_i]
            
            kernel = @cuda launch=false _kernel_compute_barrier_nonnegative(worki, zi, si, dzi, dsi, α, Cint(len_cone))
            config = launch_configuration(kernel.fun)
            threads = min(len_cone, config.threads)
            blocks = cld(len_cone, threads)
            
            CUDA.@sync kernel(worki, zi, si, dzi, dsi, α, Cint(len_cone); threads, blocks)
            barrier += sum(worki)
        else
            # For small cones, use direct GPU operations
            @views worki = work[rng_cone_i]
            @views @. worki = -logsafe_gpu((s[rng_cone_i] + α*ds[rng_cone_i])*(z[rng_cone_i] + α*dz[rng_cone_i]))
            barrier += sum(worki)
        end
    end

    return barrier
end

# # ---------------------------------------------
# # operations supported by symmetric cones only 
# # ---------------------------------------------

# # implements y = αWx + βy for the nn cone
# function mul_W!(
#     K::NonnegativeCone{T},
#     is_transpose::Symbol,
#     y::AbstractVector{T},
#     x::AbstractVector{T},
#     α::T,
#     β::T
# ) where {T}

#   #W is diagonal so ignore transposition
#   #@. y = α*(x*K.w) + β*y
#   @inbounds for i = eachindex(y)
#       y[i] = α*(x[i]*K.w[i]) + β*y[i]
#   end

#   return nothing
# end

# # implements y = αW^{-1}x + βy for the nn cone
# function mul_Winv!(
#     K::NonnegativeCone{T},
#     is_transpose::Symbol,
#     y::AbstractVector{T},
#     x::AbstractVector{T},
#     α::T,
#     β::T
# ) where {T}

#   #W is diagonal, so ignore transposition
#   #@. y = α*(x/K.w) + β.*y
#   @inbounds for i = eachindex(y)
#       y[i] = α*(x[i]/K.w[i]) + β*y[i]
#   end

#   return nothing
# end

# # implements x = λ \ z for the nn cone, where λ
# # is the internally maintained scaling variable.
# function λ_inv_circ_op!(
#     K::NonnegativeCone{T},
#     x::AbstractVector{T},
#     z::AbstractVector{T}
# ) where {T}

#     inv_circ_op!(K, x, K.λ, z)

# end

# # ---------------------------------------------
# # Jordan algebra operations for symmetric cones 
# # ---------------------------------------------

# # implements x = y ∘ z for the nn cone
# function circ_op!(
#     K::NonnegativeCone{T},
#     x::AbstractVector{T},
#     y::AbstractVector{T},
#     z::AbstractVector{T}
# ) where {T}

#     @. x = y*z

#     return nothing
# end

# # implements x = y \ z for the nn cone
# function inv_circ_op!(
#     K::NonnegativeCone{T},
#     x::AbstractVector{T},
#     y::AbstractVector{T},
#     z::AbstractVector{T}
# ) where {T}

#     @. x = z/y

#     return nothing
# end
