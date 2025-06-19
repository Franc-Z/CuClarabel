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

# -------------------------------------
# Zero Cone - 单SM优化版本
# -------------------------------------

# ========== CUDA Kernels ==========

# Kernel: 单block处理所有zero cone - 适用于小尺寸cone
function _kernel_single_block_set_zero!(
    vec::CuDeviceVector{T},
    rng_start::CuDeviceVector{Int32},
    rng_end::CuDeviceVector{Int32},
    n_cones::Int32
) where {T}
    tid = threadIdx().x
    n_threads = blockDim().x
    
    # 使用共享内存缓存范围信息（如果cone数量不大）
    # 注意：动态共享内存大小在kernel启动时指定
    if n_cones <= 64  # 对于少量cone使用共享内存
        shared_mem = @cuDynamicSharedMem(Int32, 2*n_cones)
        
        # 协作加载范围到共享内存
        if tid <= n_cones
            shared_mem[tid] = rng_start[tid]
            shared_mem[n_cones + tid] = rng_end[tid]
        end
        sync_threads()
        
        # 从共享内存读取并处理
        for cone_idx in tid:n_threads:n_cones
            start_idx = shared_mem[cone_idx]
            end_idx = shared_mem[n_cones + cone_idx]
            
            # 设置当前cone的所有元素为零
            for idx in start_idx:end_idx
                @inbounds vec[idx] = zero(T)
            end
        end
    else
        # 对于大量cone，直接从全局内存读取
        for cone_idx in tid:n_threads:n_cones
            start_idx = rng_start[cone_idx]
            end_idx = rng_end[cone_idx]
            
            # 设置当前cone的所有元素为零
            for idx in start_idx:end_idx
                @inbounds vec[idx] = zero(T)
            end
        end
    end
    
    return nothing
end

# Kernel: 单block同时处理两个向量（z和s）
function _kernel_single_block_set_zero_dual!(
    vec1::CuDeviceVector{T},
    vec2::CuDeviceVector{T},
    rng_start::CuDeviceVector{Int32},
    rng_end::CuDeviceVector{Int32},
    n_cones::Int32
) where {T}
    tid = threadIdx().x
    n_threads = blockDim().x
    
    if n_cones <= 64
        shared_mem = @cuDynamicSharedMem(Int32, 2*n_cones)
        
        # 协作加载范围到共享内存
        if tid <= n_cones
            shared_mem[tid] = rng_start[tid]
            shared_mem[n_cones + tid] = rng_end[tid]
        end
        sync_threads()
        
        # 从共享内存读取并处理
        for cone_idx in tid:n_threads:n_cones
            start_idx = shared_mem[cone_idx]
            end_idx = shared_mem[n_cones + cone_idx]
            
            # 使用向量化的内存访问模式
            for idx in start_idx:end_idx
                @inbounds begin
                    vec1[idx] = zero(T)
                    vec2[idx] = zero(T)
                end
            end
        end
    else
        # 对于大量cone，直接从全局内存读取
        for cone_idx in tid:n_threads:n_cones
            start_idx = rng_start[cone_idx]
            end_idx = rng_end[cone_idx]
            
            # 同时设置两个向量的元素为零
            for idx in start_idx:end_idx
                @inbounds begin
                    vec1[idx] = zero(T)
                    vec2[idx] = zero(T)
                end
            end
        end
    end
    
    return nothing
end

# Kernel: 带条件的单block处理（用于PrimalCone检查）
function _kernel_single_block_conditional_set_zero!(
    vec::CuDeviceVector{T},
    rng_start::CuDeviceVector{Int32},
    rng_end::CuDeviceVector{Int32},
    n_cones::Int32,
    is_primal::Bool
) where {T}
    if !is_primal
        return nothing
    end
    
    tid = threadIdx().x
    n_threads = blockDim().x
    
    # 每个线程处理多个cone
    for cone_idx in tid:n_threads:n_cones
        start_idx = rng_start[cone_idx]
        end_idx = rng_end[cone_idx]
        
        for idx in start_idx:end_idx
            @inbounds vec[idx] = zero(T)
        end
    end
    
    return nothing
end

# ========== 辅助函数 ==========

# 预处理cone范围并传输到GPU
function preprocess_cone_ranges_single_sm(rng_cones::AbstractVector, idx_list::Vector{Cint})
    n_cones = length(idx_list)
    n_cones == 0 && return (nothing, nothing, 0)
    
    ranges_start = Vector{Int32}(undef, n_cones)
    ranges_end = Vector{Int32}(undef, n_cones)
    total_elements = 0
    
    # 从GPU拷贝范围数据到CPU进行预处理
    if rng_cones isa CuArray
        rng_cones_cpu = Array(rng_cones)
        for (i, cone_idx) in enumerate(idx_list)
            rng = rng_cones_cpu[cone_idx]
            ranges_start[i] = Int32(first(rng))
            ranges_end[i] = Int32(last(rng))
            total_elements += length(rng)
        end
    else
        for (i, cone_idx) in enumerate(idx_list)
            rng = rng_cones[cone_idx]
            ranges_start[i] = Int32(first(rng))
            ranges_end[i] = Int32(last(rng))
            total_elements += length(rng)
        end
    end
    
    return CuArray(ranges_start), CuArray(ranges_end), total_elements
end

# 通用的单SM kernel启动函数
@inline function launch_single_sm_kernel!(kernel_func, vec_args, rng_cones, idx_list, extra_args...)
    isempty(idx_list) && return
    
    rng_start_gpu, rng_end_gpu, _ = preprocess_cone_ranges_single_sm(rng_cones, idx_list)
    n_cones = Int32(length(idx_list))
    
    # 动态调整线程数，确保良好的占用率
    threads = min(256, max(32, n_cones * 8))
    blocks = 1
    
    # 计算共享内存大小（对于小cone集合使用共享内存）
    shmem_size = n_cones <= 64 ? sizeof(Int32) * 2 * n_cones : 0
    
    # 启动kernel
    @cuda threads=threads blocks=blocks shmem=shmem_size kernel_func(
        vec_args..., rng_start_gpu, rng_end_gpu, n_cones, extra_args...
    )
    
    CUDA.synchronize()
end

# ========== 优化的包装函数 ==========

# place vector into zero cone
@inline function scaled_unit_shift_zero!(
    z::AbstractVector{T},
    rng_cones::AbstractVector,
    idx_eq::Vector{Cint},
    pd::PrimalOrDualCone
) where{T}
    # 只对PrimalCone设置零值
    pd == PrimalCone::PrimalOrDualCone || return
    
    launch_single_sm_kernel!(_kernel_single_block_conditional_set_zero!, (z,), rng_cones, idx_eq, true)
end

# unit initialization for asymmetric solves
@inline function unit_initialization_zero!(
    z::AbstractVector{T},
    s::AbstractVector{T},
    rng_cones::AbstractVector,
    idx_eq::Vector{Cint}
) where{T}
    launch_single_sm_kernel!(_kernel_single_block_set_zero_dual!, (z, s), rng_cones, idx_eq)
end

@inline function get_Hs_zero!(
    Hsblocks::AbstractVector{T},
    rng_blocks::AbstractVector,
    idx_eq::Vector{Cint}
) where {T}
    launch_single_sm_kernel!(_kernel_single_block_set_zero!, (Hsblocks,), rng_blocks, idx_eq)
end

# compute the product y = WᵀWx
@inline function mul_Hs_zero!(
    y::AbstractVector{T},
    rng_cones::AbstractVector,
    idx_eq::Vector{Cint}
) where {T}
    launch_single_sm_kernel!(_kernel_single_block_set_zero!, (y,), rng_cones, idx_eq)
end

@inline function affine_ds_zero!(
    ds::AbstractVector{T},
    rng_cones::AbstractVector,
    idx_eq::Vector{Cint}
) where {T}
    launch_single_sm_kernel!(_kernel_single_block_set_zero!, (ds,), rng_cones, idx_eq)
end

@inline function combined_ds_shift_zero!(
    shift::AbstractVector{T},
    rng_cones::AbstractVector,
    idx_eq::Vector{Cint}
) where {T}
    launch_single_sm_kernel!(_kernel_single_block_set_zero!, (shift,), rng_cones, idx_eq)
end

@inline function Δs_from_Δz_offset_zero!(
    out::AbstractVector{T},
    rng_cones::AbstractVector,
    idx_eq::Vector{Cint}
) where {T}
    launch_single_sm_kernel!(_kernel_single_block_set_zero!, (out,), rng_cones, idx_eq)
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

