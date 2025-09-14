# # -------------------------------------
# # Zero Cone
# # -------------------------------------

# degree(K::ZeroCone{T}) where {T} = 0
# numel(K::ZeroCone{T}) where {T}  = K.dim

# # The Zerocone reports itself as symmetric even though it is not,
# # nor does it support any of the specialised symmetric interface.
# # This cone serves as a dummy constraint to allow us to avoid 
# # implementing special handling of equalities. We want problems 
# # with both equalities and purely symmetric conic constraints to 
# # be treated as symmetric for the purposes of initialization etc 
# is_symmetric(::ZeroCone{T}) where {T} = true

# function rectify_equilibration!(
#     K::ZeroCone{T},
#     δ::AbstractVector{T},
#     e::AbstractVector{T}
# ) where{T}

#     #allow elementwise equilibration scaling
#     δ .= one(T)
#     return false
# end

# place vector into zero cone
@inline function scaled_unit_shift_zero!(
    z::AbstractVector{T},
    rng_cones::AbstractVector,
    idx_eq::Vector{Cint},
    pd::PrimalOrDualCone
) where{T}

    if pd == PrimalCone::PrimalOrDualCone #zero cone
        CUDA.@allowscalar begin
            @inbounds for i in idx_eq
                rng_cone_i = rng_cones[i]
                @views @. z[rng_cone_i] = zero(T)
            end
        end
    end
end

# unit initialization for asymmetric solves
@inline function unit_initialization_zero!(
	z::AbstractVector{T},
    s::AbstractVector{T},
    rng_cones::AbstractVector,
    idx_eq::Vector{Cint}
) where{T}

    CUDA.@allowscalar begin
        @inbounds for i in idx_eq
            rng_cone_i = rng_cones[i]
            @views @. z[rng_cone_i] = zero(T)
            @views @. s[rng_cone_i] = zero(T)
        end
    end
end

@inline function get_Hs_zero!(
    Hsblocks::AbstractVector{T},
    rng_blocks::AbstractVector,
    idx_eq::Vector{Cint}
) where {T}

    #expecting only a diagonal here, and
    #setting it to zero since this is an
    #equality condition
    CUDA.@allowscalar begin
        @inbounds for i in idx_eq
            @views @. Hsblocks[rng_blocks[i]] = zero(T)
        end
    end
end

# compute the product y = WᵀWx
@inline function mul_Hs_zero!(
    y::AbstractVector{T},
    rng_cones::AbstractVector,
    idx_eq::Vector{Cint}
) where {T}

    CUDA.@allowscalar begin
        @inbounds for i in idx_eq
            @views @. y[rng_cones[i]] = zero(T)
        end
    end
end

@inline function affine_ds_zero!(
    ds::AbstractVector{T},
    rng_cones::AbstractVector,
    idx_eq::Vector{Cint}
) where {T}

    CUDA.@allowscalar begin
        @inbounds for i in idx_eq
            @views @. ds[rng_cones[i]] = zero(T)
        end
    end
end

@inline function combined_ds_shift_zero!(
    shift::AbstractVector{T},
    rng_cones::AbstractVector,
    idx_eq::Vector{Cint}
) where {T}

    CUDA.@allowscalar begin
        @inbounds for i in idx_eq
            @views @. shift[rng_cones[i]] = zero(T)
        end
    end
end

@inline function Δs_from_Δz_offset_zero!(
    out::AbstractVector{T},
    rng_cones::AbstractVector,
    idx_eq::Vector{Cint}
) where {T}

    CUDA.@allowscalar begin
        @inbounds for i in idx_eq
            @views @. out[rng_cones[i]] = zero(T)
        end
    end
end

# function step_length(
#      K::ZeroCone{T},
#     dz::AbstractVector{T},
#     ds::AbstractVector{T},
#      z::AbstractVector{T},
#      s::AbstractVector{T},
#      settings::Settings{T},
#      αmax::T,
# ) where {T}

#     #equality constraints allow arbitrary step length
#     return (αmax,αmax)
# end

# # no compute_centrality for Zerocone
# function compute_barrier(
#     K::ZeroCone{T},
#     z::AbstractVector{T},
#     s::AbstractVector{T},
#     dz::AbstractVector{T},
#     ds::AbstractVector{T},
#     α::T
# ) where {T}

#     return zero(T)

# end

