# # ----------------------------------------------------
# # Second Order Cone
# # ----------------------------------------------------

# ===== 优化版本的辅助函数 =====
# 用于并行规约的辅助函数
@inline function warp_reduce_sum(val::T) where T
    offset = 16
    while offset > 0
        val += CUDA.shfl_down_sync(0xffffffff, val, offset)
        offset >>= 1
    end
    return val
end

@inline function block_reduce_sum(val::T, shared_mem) where T
    tid = threadIdx().x
    wid = (tid - 1) ÷ 32 + 1
    lane = (tid - 1) % 32 + 1
    
    # Warp级规约
    val = warp_reduce_sum(val)
    
    # 将每个warp的结果写入共享内存
    if lane == 1
        shared_mem[wid] = val
    end
    sync_threads()
    
    # 最后一个warp进行最终规约
    if wid == 1
        val = tid <= (blockDim().x ÷ 32) ? shared_mem[tid] : zero(T)
        val = warp_reduce_sum(val)
    end
    
    return val
end

# 智能选择器：根据cone特征判断是否使用优化版本
@inline function should_use_optimized_version(rng_cones::AbstractVector, n_shift::Cint, n_soc::Cint)
    if n_soc == 0
        return false
    end
    
    # 计算平均cone大小
    total_size = 0
    CUDA.@allowscalar for i in 1:n_soc
        total_size += length(rng_cones[i + n_shift])
    end
    avg_size = total_size ÷ n_soc
    
    # 如果cone数量少且平均大小大，使用优化版本
    return n_soc <= 8 && avg_size >= 1000
end

# 获取优化版本的线程配置
@inline function get_optimized_config(rng_cones::AbstractVector, n_shift::Cint, n_soc::Cint)
    max_cone_size = 0
    CUDA.@allowscalar for i in 1:n_soc
        max_cone_size = max(max_cone_size, length(rng_cones[i + n_shift]))
    end
    
    threads_per_cone = min(1024, max(128, nextpow(2, ceil(Int, sqrt(max_cone_size)))))
    shared_mem_size = threads_per_cone ÷ 32 + 8  # 额外空间用于存储中间结果
    
    return (threads=threads_per_cone, shared_mem_size=shared_mem_size)
end

# ===== 原始版本的kernel（保持不变） =====

function _kernel_margins_soc(
    z::AbstractVector{T},
    α::AbstractVector{T},
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_soc::Cint
) where{T}
    i = (blockIdx().x-1)*blockDim().x+threadIdx().x

    if i <= n_soc
        shift_i = i + n_shift
        rng_cone_i = rng_cones[shift_i]
        size_i = length(rng_cone_i)
        @views zi = z[rng_cone_i] 
        
        val = zero(T)
        @inbounds for j in 2:size_i 
            val += zi[j]*zi[j]
        end
        α[i]  = zi[1] - sqrt(val)
    end

    return nothing
end

# ===== 优化版本的kernel =====

# 优化版本的margin计算kernel - 使用块内并行处理每个cone
function _kernel_margins_soc_optimized(
    z::AbstractVector{T},
    α::AbstractVector{T},
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_soc::Cint
) where{T}
    # 使用二维网格：x维度是cone索引
    cone_idx = blockIdx().x
    tid = threadIdx().x
    
    # 共享内存用于规约
    shared_mem = @cuDynamicSharedMem(T, blockDim().x ÷ 32)
    
    if cone_idx <= n_soc
        shift_i = cone_idx + n_shift
        rng_cone_i = rng_cones[shift_i]
        size_i = length(rng_cone_i)
        @views zi = z[rng_cone_i]
        
        # 每个线程处理cone内的多个元素
        val = zero(T)
        # 从索引2开始，步长为blockDim().x
        idx = 2 + (tid - 1)  # 第1个线程处理索引2，第2个线程处理索引3，等等
        while idx <= size_i
            val += zi[idx] * zi[idx]
            idx += blockDim().x
        end
        
        # 块内规约求和
        val = block_reduce_sum(val, shared_mem)
        
        # 线程0计算最终结果
        if tid == 1
            α[cone_idx] = zi[1] - sqrt(val)
        end
    end
    
    return nothing
end

# ===== 单个大型SOC的特殊kernel =====

# 单个大型cone的margins计算kernel
function _kernel_margins_soc_single_cone(
    z::AbstractVector{T},
    α::AbstractVector{T},
    rng_cone,
    cone_size::Cint
) where{T}
    tid = Cint(threadIdx().x)
    bid = blockIdx().x
    
    # 共享内存用于warp规约
    shared_mem = @cuDynamicSharedMem(T, cld(blockDim().x, 32))
    
    # 每个线程计算部分和
    local_sum = zero(T)
    idx = tid + (bid - 1) * blockDim().x
    
    while idx <= cone_size - 1  # 从索引2开始
        @inbounds local_sum += z[rng_cone.start + idx] * z[rng_cone.start + idx]
        idx += gridDim().x * blockDim().x
    end
    
    # 使用优化的block规约
    warp_id = (tid - 1) ÷ 32 + 1
    lane_id = (tid - 1) % 32 + 1
    
    # Warp级规约
    local_sum = warp_reduce_sum(local_sum)
    
    # 将warp结果写入共享内存
    if lane_id == 1
        @inbounds shared_mem[warp_id] = local_sum
    end
    sync_threads()
    
    # 最终规约
    if tid <= 32
        @inbounds local_sum = tid <= cld(blockDim().x, 32) ? shared_mem[tid] : zero(T)
        local_sum = warp_reduce_sum(local_sum)
    end
    
    # 第一个线程写入结果
    if tid == 1
        if gridDim().x > 1
            # 多个块：累加部分和
            CUDA.@atomic α[1] += local_sum
        else
            # 单个块：直接计算最终结果
            @inbounds α[1] = z[rng_cone.start] - sqrt(local_sum)
        end
    end
    
    return nothing
end

# 单个cone margin计算的最终化
function _finalize_margins_soc_single_cone(
    z::AbstractVector{T},
    α::AbstractVector{T},
    rng_cone
) where{T}
    CUDA.@allowscalar begin
        val = α[1]
        α[1] = z[rng_cone.start] - sqrt(val)
    end
    return nothing
end

# ===== 统一的调用接口（智能选择版本） =====

@inline function margins_soc(
    z::AbstractVector{T},
    α::AbstractVector{T},
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_soc::Cint,
    αmin::T
) where{T}
    if n_soc == 1
        # 特殊优化：单个大型cone
        CUDA.@allowscalar begin
            rng_cone = rng_cones[n_shift + 1]
            cone_size = Cint(length(rng_cone))
        end
        
        # 初始化α[1]为0用于原子累加
        CUDA.@allowscalar α[1] = zero(T)
        
        # 使用单个cone优化kernel
        kernel = @cuda launch=false _kernel_margins_soc_single_cone(z, α, rng_cone, cone_size)
        config = launch_configuration(kernel.fun)
        threads = min(256, config.threads)
        blocks = min(256, cld(cone_size - 1, threads))  # -1因为跳过第一个元素
        
        # 分配共享内存用于warp规约
        shmem_size = cld(threads, 32) * sizeof(T)
        CUDA.@sync kernel(z, α, rng_cone, cone_size; threads, blocks, shmem=shmem_size)
        
        # 如果使用了多个块，最终化计算
        if blocks > 1
            _finalize_margins_soc_single_cone(z, α, rng_cone)
        end
        
        CUDA.@allowscalar begin
            αmin = min(αmin, α[1])
            α[1] = max(zero(T), α[1])
            return (αmin, α[1])
        end
    elseif should_use_optimized_version(rng_cones, n_shift, n_soc)
        # 使用优化版本（少量大型cone）
        config = get_optimized_config(rng_cones, n_shift, n_soc)
        shared_mem_size = sizeof(T) * config.shared_mem_size
        
        kernel = @cuda launch=false _kernel_margins_soc_optimized(z, α, rng_cones, n_shift, n_soc)
        CUDA.@sync kernel(z, α, rng_cones, n_shift, n_soc; 
                         threads=config.threads, blocks=n_soc, shmem=shared_mem_size)
        
        @views αsoc = α[1:n_soc]
        αmin = min(αmin,minimum(αsoc))
        CUDA.@sync @. αsoc = max(zero(T),αsoc)
        return (αmin, sum(αsoc))
    else
        # 使用原始版本
        kernel = @cuda launch=false _kernel_margins_soc(z, α, rng_cones, n_shift, n_soc)
        config = launch_configuration(kernel.fun)
        threads = min(n_soc, config.threads)
        blocks = cld(n_soc, threads)
        
        CUDA.@sync kernel(z, α, rng_cones, n_shift, n_soc; threads, blocks)
        
        @views αsoc = α[1:n_soc]
        αmin = min(αmin,minimum(αsoc))
        CUDA.@sync @. αsoc = max(zero(T),αsoc)
        return (αmin, sum(αsoc))
    end
end

# place vector into socone
function _kernel_scaled_unit_shift_soc!(
    z::AbstractVector{T},
    α::T,
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_soc::Cint
) where{T}

    i = (blockIdx().x-1)*blockDim().x+threadIdx().x

    if i <= n_soc
        shift_i = i + n_shift
        rng_cone_i = rng_cones[shift_i]
        @views zi = z[rng_cone_i] 
        zi[1] += α
    end

    return nothing
end

@inline function scaled_unit_shift_soc!(
    z::AbstractVector{T},
    rng_cones::AbstractVector,
    α::T,
    n_shift::Cint,
    n_soc::Cint   
) where{T}
    if n_soc == 1
        # 特殊优化：单个大型cone - 只需要给第一个元素加α
        CUDA.@allowscalar begin
            rng_cone = rng_cones[n_shift + 1]
            z[rng_cone.start] += α
        end
    else
        # 原始实现：多个cone
        kernel = @cuda launch=false _kernel_scaled_unit_shift_soc!(z, α, rng_cones, n_shift, n_soc)
        config = launch_configuration(kernel.fun)
        threads = min(n_soc, config.threads)
        blocks = cld(n_soc, threads)

        CUDA.@sync kernel(z, α, rng_cones, n_shift, n_soc; threads, blocks)
    end
end

# 单个大型cone的unit initialization kernel
function _kernel_unit_initialization_soc_single_cone!(
    z::AbstractVector{T},
    s::AbstractVector{T},
    rng_cone,
    cone_size::Cint
) where{T}
    idx = (blockIdx().x-1)*blockDim().x+threadIdx().x
    
    if idx == 1
        @inbounds z[rng_cone.start] = one(T)
        @inbounds s[rng_cone.start] = one(T)
    elseif idx <= cone_size
        @inbounds z[rng_cone.start + idx - 1] = zero(T)
        @inbounds s[rng_cone.start + idx - 1] = zero(T)
    end
    
    return nothing
end

# unit initialization for asymmetric solves
function _kernel_unit_initialization_soc!(
    z::AbstractVector{T},
    s::AbstractVector{T},
    rng_cones::AbstractVector,
    n_linear::Cint,
    n_soc::Cint
) where{T}

    i = (blockIdx().x-1)*blockDim().x+threadIdx().x

    if i <= n_soc
        shift_i = i + n_linear
        rng_cone_i = rng_cones[shift_i]
        @views zi = z[rng_cone_i] 
        @views si = s[rng_cone_i] 
        zi[1] = one(T)
        @inbounds for j in 2:length(zi)
            zi[j] = zero(T)
        end

        si[1] = one(T)
        @inbounds for j in 2:length(si)
            si[j] = zero(T)
        end
    end
 
    return nothing
end 

# 优化版本的unit initialization
function _kernel_unit_initialization_soc_optimized!(
    z::AbstractVector{T},
    s::AbstractVector{T},
    rng_cones::AbstractVector,
    n_linear::Cint,
    n_soc::Cint
) where{T}
    cone_idx = blockIdx().x
    tid = threadIdx().x
    
    if cone_idx <= n_soc
        shift_i = cone_idx + n_linear
        rng_cone_i = rng_cones[shift_i]
        size_i = length(rng_cone_i)
        @views zi = z[rng_cone_i]
        @views si = s[rng_cone_i]
        
        # 并行初始化
        if tid == 1
            zi[1] = one(T)
            si[1] = one(T)
        end
        
        # 其他线程处理剩余元素
        idx = tid + 1
        while idx <= size_i
            if idx > 1
                zi[idx] = zero(T)
                si[idx] = zero(T)
            end
            idx += blockDim().x
        end
    end
    
    return nothing
end

@inline function unit_initialization_soc!(
    z::AbstractVector{T},
    s::AbstractVector{T},
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_soc::Cint
) where{T}
    if n_soc == 1
        # 特殊优化：单个大型cone
        CUDA.@allowscalar begin
            rng_cone = rng_cones[n_shift + 1]
            cone_size = Cint(length(rng_cone))
        end
        
        kernel = @cuda launch=false _kernel_unit_initialization_soc_single_cone!(z, s, rng_cone, cone_size)
        config = launch_configuration(kernel.fun)
        threads = min(cone_size, config.threads)
        blocks = cld(cone_size, threads)
        
        CUDA.@sync kernel(z, s, rng_cone, cone_size; threads, blocks)
    elseif should_use_optimized_version(rng_cones, n_shift, n_soc)
        # 使用优化版本（少量大型cone）
        config = get_optimized_config(rng_cones, n_shift, n_soc)
        kernel = @cuda launch=false _kernel_unit_initialization_soc_optimized!(z, s, rng_cones, n_shift, n_soc)
        CUDA.@sync kernel(z, s, rng_cones, n_shift, n_soc; threads=config.threads, blocks=n_soc)
    else
        # 使用原始版本
        kernel = @cuda launch=false _kernel_unit_initialization_soc!(z, s, rng_cones, n_shift, n_soc)
        config = launch_configuration(kernel.fun)
        threads = min(n_soc, config.threads)
        blocks = cld(n_soc, threads)
        
        CUDA.@sync kernel(z, s, rng_cones, n_shift, n_soc; threads, blocks)
    end
end

# 单个大型cone的identity scaling kernel
function _kernel_set_identity_scaling_soc_single_cone!(
    w::AbstractVector{T},
    η::AbstractVector{T},
    rng_cone,
    cone_size::Cint
) where{T}
    idx = (blockIdx().x-1)*blockDim().x+threadIdx().x
    
    if idx == 1
        @inbounds w[rng_cone.start] = one(T)
        @inbounds η[1] = one(T)
    elseif idx <= cone_size
        @inbounds w[rng_cone.start + idx - 1] = zero(T)
    end
    
    return nothing
end

# # configure cone internals to provide W = I scaling
function _kernel_set_identity_scaling_soc!(
    w::AbstractVector{T},
    η::AbstractVector{T},
    rng_cones::AbstractVector,
    n_linear::Cint,
    n_soc::Cint
) where {T}

    i = (blockIdx().x-1)*blockDim().x+threadIdx().x

    if i <= n_soc
        shift_i = i + n_linear
        rng_cone_i = rng_cones[shift_i]
        size_i = length(rng_cone_i)
        @views wi = w[rng_cone_i] 
        wi[1] = one(T)
        @inbounds for j in 2:size_i 
            wi[j] = zero(T)
        end
        η[i]  = one(T)
    end

    return nothing
end

@inline function set_identity_scaling_soc!(
    w::AbstractVector{T},
    η::AbstractVector{T},
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_soc::Cint
) where {T}
    if n_soc == 1
        # 特殊优化：单个大型cone
        CUDA.@allowscalar begin
            rng_cone = rng_cones[n_shift + 1]
            cone_size = Cint(length(rng_cone))
        end
        
        kernel = @cuda launch=false _kernel_set_identity_scaling_soc_single_cone!(w, η, rng_cone, cone_size)
        config = launch_configuration(kernel.fun)
        threads = min(cone_size, config.threads)
        blocks = cld(cone_size, threads)
        
        CUDA.@sync kernel(w, η, rng_cone, cone_size; threads, blocks)
    else
        # 原始实现：多个cone
        kernel = @cuda launch=false _kernel_set_identity_scaling_soc!(w, η, rng_cones, n_shift, n_soc)
        config = launch_configuration(kernel.fun)
        threads = min(n_soc, config.threads)
        blocks = cld(n_soc, threads)

        CUDA.@sync kernel(w, η, rng_cones, n_shift, n_soc; threads, blocks)
    end
end

@inline function set_identity_scaling_soc_sparse!(
    d::AbstractVector{T},
    vut::AbstractVector{T},
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_sparse_soc::Cint
) where {T}
    fill!(vut, 0)

    shift = 1
    CUDA.@allowscalar for i in 1:n_sparse_soc
        d[i]  = T(0.5)
        len_i = length(rng_cones[i + n_shift])
        vut[shift+len_i] = sqrt(T(0.5))
        shift += 2*len_i
    end
end

function _kernel_update_scaling_soc!(
    s::AbstractVector{T},
    z::AbstractVector{T},
    w::AbstractVector{T},
    λ::AbstractVector{T},
    η::AbstractVector{T},
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_soc::Cint
) where {T}

    i = (blockIdx().x-1)*blockDim().x+threadIdx().x

    if i <= n_soc
        shift_i = i + n_shift
        rng_i = rng_cones[shift_i]
        @views zi = z[rng_i] 
        @views si = s[rng_i] 
        @views wi = w[rng_i] 
        @views λi = λ[rng_i]

        #first calculate the scaled vector w
        @views zscale = _sqrt_soc_residual_gpu(zi)
        @views sscale = _sqrt_soc_residual_gpu(si)

        #the leading scalar term for W^TW
        η[i] = sqrt(sscale/zscale)

        # construct w and normalize
        @inbounds for k in rng_i
            w[k] = s[k]/(sscale)
        end

        wi[1]  += zi[1]/(zscale)

        @inbounds for j in 2:length(wi)
            wi[j] -= zi[j]/(zscale)
        end
    
        wscale = _sqrt_soc_residual_gpu(wi)
        wi ./= wscale

        #try to force badly scaled w to come out normalized
        w1sq = zero(T)
        @inbounds for j in 2:length(wi)
            w1sq += wi[j]*wi[j]
        end
        wi[1] = sqrt(1 + w1sq)

        #Compute the scaling point λ.   Should satisfy λ = Wz = W^{-T}s
        γi = 0.5 * wscale
        λi[1] = γi 

        coef = inv(si[1]/sscale + zi[1]/zscale + 2*γi)
        c1 = ((γi + zi[1]/zscale)/sscale)
        c2 = ((γi + si[1]/sscale)/zscale)
        @inbounds for j in 2:length(λi)
            λi[j] = coef*(c1*si[j] +c2*zi[j])
        end
        λi .*= sqrt(sscale*zscale)
    end

    return nothing
end

# 优化版本的update scaling kernel
function _kernel_update_scaling_soc_optimized!(
    s::AbstractVector{T},
    z::AbstractVector{T},
    w::AbstractVector{T},
    λ::AbstractVector{T},
    η::AbstractVector{T},
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_soc::Cint
) where {T}
    cone_idx = blockIdx().x
    tid = threadIdx().x
    
    # 共享内存分配
    shared_size = blockDim().x ÷ 32 + 4  # 额外空间存储中间结果
    shared_mem = @cuDynamicSharedMem(T, shared_size)
    
    if cone_idx <= n_soc
        shift_i = cone_idx + n_shift
        rng_i = rng_cones[shift_i]
        size_i = length(rng_i)
        @views zi = z[rng_i]
        @views si = s[rng_i]
        @views wi = w[rng_i]
        @views λi = λ[rng_i]
        
        # 并行计算z的residual = z[1]^2 - sum(z[2:end].^2)
        val_z = zero(T)
        # 每个线程计算部分元素
        idx = tid
        while idx <= size_i
            if idx == 1
                val_z += zi[idx] * zi[idx]  # 第1个元素是正的
            else
                val_z -= zi[idx] * zi[idx]  # 其余元素是负的
            end
            idx += blockDim().x
        end
        val_z = block_reduce_sum(val_z, shared_mem)
        
        sync_threads()
        
        # 存储zscale到共享内存
        if tid == 1
            shared_mem[end-3] = val_z > 0 ? sqrt(val_z) : zero(T)
        end
        sync_threads()
        zscale = shared_mem[end-3]
        
        # 并行计算s的residual = s[1]^2 - sum(s[2:end].^2)
        val_s = zero(T)
        idx = tid
        while idx <= size_i
            if idx == 1
                val_s += si[idx] * si[idx]  # 第1个元素是正的
            else
                val_s -= si[idx] * si[idx]  # 其余元素是负的
            end
            idx += blockDim().x
        end
        val_s = block_reduce_sum(val_s, shared_mem)
        
        sync_threads()
        
        # 存储sscale到共享内存
        if tid == 1
            shared_mem[end-2] = val_s > 0 ? sqrt(val_s) : zero(T)
        end
        sync_threads()
        sscale = shared_mem[end-2]
        
        # 设置η
        if tid == 1
            η[cone_idx] = sqrt(sscale/zscale)
        end
        
        # 并行构建w
        idx = tid
        while idx <= size_i
            if idx == 1
                w[rng_i[idx]] = s[rng_i[idx]]/sscale + zi[1]/zscale
            else
                w[rng_i[idx]] = s[rng_i[idx]]/sscale - zi[idx]/zscale
            end
            idx += blockDim().x
        end
        
        sync_threads()
        
        # 计算w的residual用于归一化
        val_w = zero(T)
        idx = tid
        while idx <= size_i
            if idx == 1
                val_w += wi[idx] * wi[idx]
            else
                val_w -= wi[idx] * wi[idx]
            end
            idx += blockDim().x
        end
        val_w = block_reduce_sum(val_w, shared_mem)
        
        sync_threads()
        
        if tid == 1
            shared_mem[end-1] = val_w > 0 ? sqrt(val_w) : zero(T)
        end
        sync_threads()
        wscale = shared_mem[end-1]
        
        # 并行归一化w
        idx = tid
        while idx <= size_i
            wi[idx] /= wscale
            idx += blockDim().x
        end
        
        sync_threads()
        
        # 强制归一化（计算w1sq = sum(w[2:end].^2)）
        val_w1sq = zero(T)
        idx = 2 + (tid - 1)  # 从索引2开始
        while idx <= size_i
            val_w1sq += wi[idx] * wi[idx]
            idx += blockDim().x
        end
        val_w1sq = block_reduce_sum(val_w1sq, shared_mem)
        
        if tid == 1
            wi[1] = sqrt(1 + val_w1sq)
            shared_mem[end] = 0.5 * wscale  # γi
        end
        sync_threads()
        
        γi = shared_mem[end]
        
        # 计算λ
        if tid == 1
            λi[1] = γi
            coef = inv(si[1]/sscale + zi[1]/zscale + 2*γi)
            c1 = ((γi + zi[1]/zscale)/sscale)
            c2 = ((γi + si[1]/sscale)/zscale)
            # 存储到共享内存供其他线程使用
            shared_mem[end-3] = coef
            shared_mem[end-2] = c1
            shared_mem[end-1] = c2
        end
        sync_threads()
        
        coef = shared_mem[end-3]
        c1 = shared_mem[end-2]
        c2 = shared_mem[end-1]
        
        # 并行计算λ的其余元素
        idx = tid + 1
        while idx <= size_i
            λi[idx] = coef * (c1 * si[idx] + c2 * zi[idx])
            idx += blockDim().x
        end
        
        sync_threads()
        
        # 最终缩放λ
        scale_factor = sqrt(sscale * zscale)
        idx = tid
        while idx <= size_i
            λi[idx] *= scale_factor
            idx += blockDim().x
        end
    end
    
    return nothing
end

@inline function update_scaling_soc!(
    s::AbstractVector{T},
    z::AbstractVector{T},
    w::AbstractVector{T},
    λ::AbstractVector{T},
    η::AbstractVector{T},
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_soc::Cint
) where {T}
    #=
    if n_soc == 1
        # 特殊优化：单个大型cone - 使用与原始版本相同的kernel，但只处理一个cone
        # 这避免了@allowscalar的性能问题
        kernel = @cuda launch=false _kernel_update_scaling_soc!(s, z, w, λ, η, rng_cones, n_shift, n_soc)
        config = launch_configuration(kernel.fun)
        # 对于单个cone，只需要一个线程块
        threads = 1
        blocks = 1
        
        CUDA.@sync kernel(s, z, w, λ, η, rng_cones, n_shift, n_soc; threads, blocks)
    else
    =#
    if should_use_optimized_version(rng_cones, n_shift, n_soc)
        # 使用优化版本（少量大型cone）
        config = get_optimized_config(rng_cones, n_shift, n_soc)
        shared_mem_size = sizeof(T) * config.shared_mem_size
        
        kernel = @cuda launch=false _kernel_update_scaling_soc_optimized!(s, z, w, λ, η, rng_cones, n_shift, n_soc)
        CUDA.@sync kernel(s, z, w, λ, η, rng_cones, n_shift, n_soc; 
                         threads=config.threads, blocks=n_soc, shmem=shared_mem_size)
    else
        # 使用原始版本
        kernel = @cuda launch=false _kernel_update_scaling_soc!(s, z, w, λ, η, rng_cones, n_shift, n_soc)
        config = launch_configuration(kernel.fun)
        threads = min(n_soc, config.threads)
        blocks = cld(n_soc, threads)
        
        CUDA.@sync kernel(s, z, w, λ, η, rng_cones, n_shift, n_soc; threads, blocks)
    end

end


@inline function _update_scaling_soc_sparse!(
    w::AbstractVector{T},
    u::AbstractVector{T},
    v::AbstractVector{T},
    η::T
) where {T}

    #Populate sparse expansion terms if allocated
    #various intermediate calcs for u,v,d
    α  = 2*w[1]

    #Scalar d is the upper LH corner of the diagonal
    #term in the rank-2 update form of W^TW
    
    wsq    = 2*w[1]*w[1] - 1
    wsqinv = 1/wsq
    d    = wsqinv / 2

    #the vectors for the rank two update
    #representation of W^TW
    u0  = sqrt(wsq - d)
    u1 = α/u0
    v0 = zero(T)
    v1 = sqrt( 2*(2 + wsqinv)/(2*wsq - wsqinv))
    
    minus_η2 = -η*η 
    u[1] = minus_η2*u0
    @views u[2:end] .= minus_η2.*u1.*w[2:end]
    v[1] = minus_η2*v0
    @views v[2:end] .= minus_η2.*v1.*w[2:end]
    CUDA.synchronize()

    return d
end

@inline function update_scaling_soc_sparse_sequential!(
    w::AbstractVector{T},
    η::AbstractVector{T},
    d::AbstractVector{T},
    vut::AbstractVector{T},
    rng_cones::AbstractVector,
    numel_linear::Cint,
    n_shift::Cint,
    n_sparse_soc::Cint
) where {T}
    CUDA.@allowscalar for i in 1:n_sparse_soc
        shift_i = i + n_shift
        rng_i = rng_cones[shift_i]
        len_i = length(rng_i)
        rng_sparse_i = rng_i .- numel_linear
        startidx = 2*(rng_sparse_i.stop - len_i)
        wi = view(w, rng_i)
        vi = view(vut, (startidx+1):(startidx+len_i))
        ui = view(vut, (startidx+len_i+1):(startidx+2*len_i))
        ηi = η[i]

        d[i] = _update_scaling_soc_sparse!(wi,ui,vi,ηi)
    end
end

@inline function _kernel_update_scaling_soc_sparse_parallel!(
    w::AbstractVector{T},
    η::AbstractVector{T},
    d::AbstractVector{T},
    vut::AbstractVector{T},
    rng_cones::AbstractVector,
    numel_linear::Cint,
    n_shift::Cint,
    n_sparse_soc::Cint
) where {T}
    
    i = (blockIdx().x-1)*blockDim().x+threadIdx().x

    if i <= n_sparse_soc 
        shift_i = i + n_shift
        rng_i = rng_cones[shift_i]
        len_i = length(rng_i)
        rng_sparse_i = rng_i .- numel_linear
        startidx = 2*(rng_sparse_i.stop - len_i)
        wi = view(w, rng_i)
        vi = view(vut, (startidx+1):(startidx+len_i))
        ui = view(vut, (startidx+len_i+1):(startidx+2*len_i))

        #Unroll function _update_scaling_soc_sparse!()
        #Populate sparse expansion terms if allocated
        #various intermediate calcs for u,v,d
        α  = 2*wi[1]

        #Scalar d is the upper LH corner of the diagonal
        #term in the rank-2 update form of W^TW
        
        wsq    = 2*wi[1]*wi[1] - 1
        wsqinv = 1/wsq
        di    = wsqinv / 2
        d[i]   = di

        #the vectors for the rank two update
        #representation of W^TW
        u0  = sqrt(wsq - di)
        u1 = α/u0
        v0 = zero(T)
        v1 = sqrt( 2*(2 + wsqinv)/(2*wsq - wsqinv))
        
        minus_η2 = -η[i]*η[i]
        ui[1] = minus_η2*u0
        vi[1] = minus_η2*v0

        @inbounds for j in 2:length(ui)
            ui[j] = minus_η2*u1*wi[j]
            vi[j] = minus_η2*v1*wi[j]
        end
    end
end

@inline function update_scaling_soc_sparse_parallel!(
    w::AbstractVector{T},
    η::AbstractVector{T},
    d::AbstractVector{T},
    vut::AbstractVector{T},
    rng_cones::AbstractVector,
    numel_linear::Cint,
    n_shift::Cint,
    n_sparse_soc::Cint
) where {T}

    kernel = @cuda launch=false _kernel_update_scaling_soc_sparse_parallel!(w, η, d, vut, rng_cones, numel_linear, n_shift, n_sparse_soc)
    config = launch_configuration(kernel.fun)
    threads = min(n_sparse_soc, config.threads)
    blocks = cld(n_sparse_soc, threads)

    CUDA.@sync kernel(w, η, d, vut, rng_cones, numel_linear, n_shift, n_sparse_soc; threads, blocks)

end

function _kernel_get_Hs_soc_dense!(
    Hsblocks::AbstractVector{T},
    w::AbstractVector{T},
    η::AbstractVector{T},
    rng_cones::AbstractVector,
    rng_blocks::AbstractVector,
    n_shift::Cint,
    n_sparse_soc::Cint,
    n_dense_soc::Cint
) where {T}

    i = (blockIdx().x-1)*blockDim().x+threadIdx().x

    if i <= n_dense_soc
        shift_i = i + n_shift
        rng_cone_i = rng_cones[shift_i]
        rng_block_i = rng_blocks[shift_i]
        size_i = length(rng_cone_i)
        @views wi = w[rng_cone_i] 
        @views Hsblocki = Hsblocks[rng_block_i]

        hidx = one(Cint)
        @inbounds for col in rng_cone_i
            wcol = w[col]
            @inbounds for row in rng_cone_i
                Hsblocki[hidx] = 2*w[row]*wcol
                hidx += 1
            end 
        end
        Hsblocki[1] -= one(T)
        @inbounds for ind in 2:size_i
            Hsblocki[(ind-1)*size_i + ind] += one(T)
        end
        Hsblocki .*= η[n_sparse_soc+i]^2
    end

    return nothing
end

@inline function get_Hs_soc_dense!(
    Hsblocks::AbstractVector{T},
    w::AbstractVector{T},
    η::AbstractVector{T},
    rng_cones::AbstractVector,
    rng_blocks::AbstractVector,
    n_shift::Cint,
    n_sparse_soc::Cint,
    n_dense_soc::Cint
) where {T}

    kernel = @cuda launch=false _kernel_get_Hs_soc_dense!(Hsblocks, w, η, rng_cones, rng_blocks, n_shift, n_sparse_soc, n_dense_soc)
    config = launch_configuration(kernel.fun)
    threads = min(n_dense_soc, config.threads)
    blocks = cld(n_dense_soc, threads)

    CUDA.@sync kernel(Hsblocks, w, η, rng_cones, rng_blocks, n_shift, n_sparse_soc, n_dense_soc; threads, blocks)

end

@inline function get_Hs_soc_sparse_sequential!(
    Hsblocks::AbstractVector{T},
    η::AbstractVector{T},
    d::AbstractVector{T},
    rng_blocks::AbstractVector,
    n_shift::Cint,
    n_sparse_soc::Cint
) where {T}

    #For sparse form, we are returning here the diagonal D block 
    #from the sparse representation of W^TW, but not the
    #extra two entries at the bottom right of the block.
    #The AbstractVector for s and z (and its views) don't
    #know anything about the 2 extra sparsifying entries
    CUDA.@allowscalar for i in 1:n_sparse_soc
        shift_i = i + n_shift
        rng_block_i = rng_blocks[shift_i]
        Hsblock_i = view(Hsblocks, rng_block_i)
        CUDA.@sync @. Hsblock_i = η[i]^2
        Hsblock_i[1] *= d[i]
    end
end

function _kernel_get_Hs_soc_sparse_parallel!(
    Hsblocks::AbstractVector{T},
    η::AbstractVector{T},
    d::AbstractVector{T},
    rng_blocks::AbstractVector,
    n_shift::Cint,
    n_sparse_soc::Cint
) where {T}

    i = (blockIdx().x-1)*blockDim().x+threadIdx().x

    if i <= n_sparse_soc
        shift_i = i + n_shift
        rng_block_i = rng_blocks[shift_i]

        η2 = η[i]^2
        @inbounds for col in rng_block_i
            Hsblocks[col] = η2
        end
        Hsblocks[rng_block_i.start] *= d[i]

    end

    return nothing
end

@inline function get_Hs_soc_sparse_parallel!(
    Hsblocks::AbstractVector{T},
    η::AbstractVector{T},
    d::AbstractVector{T},
    rng_blocks::AbstractVector,
    n_shift::Cint,
    n_sparse_soc::Cint
) where {T}

    #For sparse form, we are returning here the diagonal D block 
    #from the sparse representation of W^TW, but not the
    #extra two entries at the bottom right of the block.
    #The AbstractVector for s and z (and its views) don't
    #know anything about the 2 extra sparsifying entries
    kernel = @cuda launch=false _kernel_get_Hs_soc_sparse_parallel!(Hsblocks, η, d, rng_blocks, n_shift, n_sparse_soc)
    config = launch_configuration(kernel.fun)
    threads = min(n_sparse_soc, config.threads)
    blocks = cld(n_sparse_soc, threads)

    CUDA.@sync kernel(Hsblocks, η, d, rng_blocks, n_shift, n_sparse_soc; threads, blocks)

end

# compute the product y = WᵀWx
function _kernel_mul_Hs_soc!(
    y::AbstractVector{T},
    x::AbstractVector{T},
    w::AbstractVector{T},
    η::AbstractVector{T},
    rng_cones::AbstractVector,
    n_linear::Cint,
    n_soc::Cint
) where {T}

    # y = = H^{-1}x = W^TWx
    # where H^{-1} = \eta^{2} (2*ww^T - J)
    i = (blockIdx().x-1)*blockDim().x+threadIdx().x

    if i <= n_soc
        shift_i = i + n_linear
        rng_cone_i = rng_cones[shift_i]
        size_i = length(rng_cone_i)
        @views xi = x[rng_cone_i] 
        @views yi = y[rng_cone_i] 
        @views wi = w[rng_cone_i] 

        c = 2*_dot_xy_gpu(wi,xi,1:size_i)

        yi[1] = -xi[1] + c*wi[1]
        @inbounds for j in 2:size_i
            yi[j] = xi[j] + c*wi[j]
        end

        _multiply_gpu(yi,η[i]^2)
    end

    return nothing
end

# 优化版本的矩阵向量乘法
function _kernel_mul_Hs_soc_optimized!(
    y::AbstractVector{T},
    x::AbstractVector{T},
    w::AbstractVector{T},
    η::AbstractVector{T},
    rng_cones::AbstractVector,
    n_linear::Cint,
    n_soc::Cint
) where {T}
    cone_idx = blockIdx().x
    tid = threadIdx().x
    
    shared_mem = @cuDynamicSharedMem(T, blockDim().x ÷ 32 + 1)
    
    if cone_idx <= n_soc
        shift_i = cone_idx + n_linear
        rng_cone_i = rng_cones[shift_i]
        size_i = length(rng_cone_i)
        @views xi = x[rng_cone_i]
        @views yi = y[rng_cone_i]
        @views wi = w[rng_cone_i]
        
        # 并行计算点积 w·x
        val = zero(T)
        idx = tid
        while idx <= size_i
            val += wi[idx] * xi[idx]
            idx += blockDim().x
        end
        val = block_reduce_sum(val, shared_mem)
        
        if tid == 1
            shared_mem[end] = 2 * val
        end
        sync_threads()
        
        c = shared_mem[end]
        eta_sq = η[cone_idx]^2
        
        # 并行计算输出
        idx = tid
        while idx <= size_i
            if idx == 1
                yi[idx] = eta_sq * (-xi[idx] + c * wi[idx])
            else
                yi[idx] = eta_sq * (xi[idx] + c * wi[idx])
            end
            idx += blockDim().x
        end
    end
    
    return nothing
end

@inline function mul_Hs_soc!(
    y::AbstractVector{T},
    x::AbstractVector{T},
    w::AbstractVector{T},
    η::AbstractVector{T},
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_soc::Cint
) where {T}
    if n_soc == 1
        # 特殊优化：单个大型cone - 使用原始kernel
        kernel = @cuda launch=false _kernel_mul_Hs_soc!(y, x, w, η, rng_cones, n_shift, n_soc)
        config = launch_configuration(kernel.fun)
        threads = 1
        blocks = 1
        
        CUDA.@sync kernel(y, x, w, η, rng_cones, n_shift, n_soc; threads, blocks)
    elseif should_use_optimized_version(rng_cones, n_shift, n_soc)
        # 使用优化版本（少量大型cone）
        config = get_optimized_config(rng_cones, n_shift, n_soc)
        shared_mem_size = sizeof(T) * (config.threads ÷ 32 + 1)
        
        kernel = @cuda launch=false _kernel_mul_Hs_soc_optimized!(y, x, w, η, rng_cones, n_shift, n_soc)
        CUDA.@sync kernel(y, x, w, η, rng_cones, n_shift, n_soc; 
                         threads=config.threads, blocks=n_soc, shmem=shared_mem_size)
    else
        # 使用原始版本
        kernel = @cuda launch=false _kernel_mul_Hs_soc!(y, x, w, η, rng_cones, n_shift, n_soc)
        config = launch_configuration(kernel.fun)
        threads = min(n_soc, config.threads)
        blocks = cld(n_soc, threads)
        
        CUDA.@sync kernel(y, x, w, η, rng_cones, n_shift, n_soc; threads, blocks)
    end
end

@inline function mul_Hs_dense_soc!(
    y::AbstractVector{T},
    x::AbstractVector{T},
    w::AbstractVector{T},
    η::AbstractVector{T},
    rng_cones::AbstractVector,
    n_linear::Cint,
    n_sparse_soc::Cint,
    n_soc::Cint
) where {T}

    n_shift = n_linear + n_sparse_soc
    n_soc_dense = n_soc - n_sparse_soc
    η_shift = view(η, (n_sparse_soc+1):n_soc)

    kernel = @cuda launch=false _kernel_mul_Hs_soc!(y, x, w, η_shift, rng_cones, n_shift, n_soc_dense)
    config = launch_configuration(kernel.fun)
    threads = min(n_soc, config.threads)
    blocks = cld(n_soc, threads)

    CUDA.@sync kernel(y, x, w, η_shift, rng_cones, n_shift, n_soc_dense; threads, blocks)
end

# returns x = λ ∘ λ for the socone
function _kernel_affine_ds_soc!(
    ds::AbstractVector{T},
    λ::AbstractVector{T},
    rng_cones::AbstractVector,
    n_linear::Cint,
    n_soc::Cint
) where {T}

    i = (blockIdx().x-1)*blockDim().x+threadIdx().x

    if i <= n_soc
        shift_i = i + n_linear
        rng_cone_i = rng_cones[shift_i]
        size_i = length(rng_cone_i)
        @views dsi = ds[rng_cone_i] 
        @views λi = λ[rng_cone_i] 

        #circ product λ∘λ
        dsi[1] = zero(T)
        for j in 1:length(dsi)
            dsi[1] += λi[j]*λi[j]
        end
        λi0 = λi[1]
        for j = 2:length(dsi)
            dsi[j] = 2*λi0*λi[j]
        end
      
    end

    return nothing

end

# 优化版本的仿射变换
function _kernel_affine_ds_soc_optimized!(
    ds::AbstractVector{T},
    λ::AbstractVector{T},
    rng_cones::AbstractVector,
    n_linear::Cint,
    n_soc::Cint
) where {T}
    cone_idx = blockIdx().x
    tid = threadIdx().x
    
    shared_mem = @cuDynamicSharedMem(T, blockDim().x ÷ 32 + 1)
    
    if cone_idx <= n_soc
        shift_i = cone_idx + n_linear
        rng_cone_i = rng_cones[shift_i]
        size_i = length(rng_cone_i)
        @views dsi = ds[rng_cone_i]
        @views λi = λ[rng_cone_i]
        
        # 并行计算 λ·λ
        val = zero(T)
        idx = tid
        while idx <= size_i
            val += λi[idx] * λi[idx]
            idx += blockDim().x
        end
        val = block_reduce_sum(val, shared_mem)
        
        if tid == 1
            dsi[1] = val
            shared_mem[end] = λi[1]
        end
        sync_threads()
        
        λi0 = shared_mem[end]
        
        # 并行计算其余元素
        idx = tid + 1
        while idx <= size_i
            dsi[idx] = 2 * λi0 * λi[idx]
            idx += blockDim().x
        end
    end
    
    return nothing
end

@inline function affine_ds_soc!(
    ds::AbstractVector{T},
    λ::AbstractVector{T},
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_soc::Cint
) where {T}
    if n_soc == 1
        # 特殊优化：单个大型cone - 使用原始kernel
        kernel = @cuda launch=false _kernel_affine_ds_soc!(ds, λ, rng_cones, n_shift, n_soc)
        config = launch_configuration(kernel.fun)
        threads = 1
        blocks = 1
        
        CUDA.@sync kernel(ds, λ, rng_cones, n_shift, n_soc; threads, blocks)
    elseif should_use_optimized_version(rng_cones, n_shift, n_soc)
        # 使用优化版本（少量大型cone）
        config = get_optimized_config(rng_cones, n_shift, n_soc)
        shared_mem_size = sizeof(T) * (config.threads ÷ 32 + 1)
        
        kernel = @cuda launch=false _kernel_affine_ds_soc_optimized!(ds, λ, rng_cones, n_shift, n_soc)
        CUDA.@sync kernel(ds, λ, rng_cones, n_shift, n_soc; 
                         threads=config.threads, blocks=n_soc, shmem=shared_mem_size)
    else
        # 使用原始版本
        kernel = @cuda launch=false _kernel_affine_ds_soc!(ds, λ, rng_cones, n_shift, n_soc)
        config = launch_configuration(kernel.fun)
        threads = min(n_soc, config.threads)
        blocks = cld(n_soc, threads)
        
        CUDA.@sync kernel(ds, λ, rng_cones, n_shift, n_soc; threads, blocks)
    end
end

# 优化版本：块内并行处理每个cone
function _kernel_combined_ds_shift_soc_optimized!(
    shift::AbstractVector{T},
    step_z::AbstractVector{T},
    step_s::AbstractVector{T},
    w::AbstractVector{T},
    η::AbstractVector{T},
    rng_cones::AbstractVector,
    n_linear::Cint,
    n_soc::Cint,
    σμ::T
) where {T}
    cone_idx = blockIdx().x
    tid = threadIdx().x
    
    # 共享内存分配
    shared_size = blockDim().x ÷ 32 + 8  # 用于规约和存储中间结果
    shared_mem = @cuDynamicSharedMem(T, shared_size)
    
    if cone_idx <= n_soc
        shift_i = cone_idx + n_linear
        rng_cone_i = rng_cones[shift_i]
        size_i = length(rng_cone_i)
        @views step_zi = step_z[rng_cone_i]
        @views step_si = step_s[rng_cone_i]
        @views wi = w[rng_cone_i]
        @views shifti = shift[rng_cone_i]
        
        # 先复制数据到shift作为临时空间
        idx = tid
        while idx <= size_i
            shifti[idx] = step_zi[idx]
            idx += blockDim().x
        end
        sync_threads()
        
        # Step 1: 计算 WΔz
        # 计算 ζ = Σ(w[j]*tmp[j]) for j=2:end
        val_ζ = zero(T)
        idx = 2 + (tid - 1)
        while idx <= size_i
            val_ζ += wi[idx] * shifti[idx]
            idx += blockDim().x
        end
        val_ζ = block_reduce_sum(val_ζ, shared_mem)
        
        if tid == 1
            shared_mem[end-7] = val_ζ
            shared_mem[end-6] = η[cone_idx]
            shared_mem[end-5] = one(T) / (one(T) + wi[1])
            shared_mem[end-4] = shifti[1] + val_ζ / (one(T) + wi[1])  # c for Δz
        end
        sync_threads()
        
        ζ_z = shared_mem[end-7]
        η_val = shared_mem[end-6]
        inv_1_plus_w1 = shared_mem[end-5]
        c_z = shared_mem[end-4]
        
        # 更新 step_z
        idx = tid
        while idx <= size_i
            if idx == 1
                step_zi[idx] = η_val * (wi[1] * shifti[1] + ζ_z)
            else
                step_zi[idx] = η_val * (shifti[idx] + c_z * wi[idx])
            end
            idx += blockDim().x
        end
        sync_threads()
        
        # Step 2: 复制step_s到临时空间并计算 W⁻¹Δs
        idx = tid
        while idx <= size_i
            shifti[idx] = step_si[idx]
            idx += blockDim().x
        end
        sync_threads()
        
        # 计算 ζ = Σ(w[j]*tmp[j]) for j=2:end
        val_ζ = zero(T)
        idx = 2 + (tid - 1)
        while idx <= size_i
            val_ζ += wi[idx] * shifti[idx]
            idx += blockDim().x
        end
        val_ζ = block_reduce_sum(val_ζ, shared_mem)
        
        if tid == 1
            shared_mem[end-3] = val_ζ  # ζ_s
            shared_mem[end-2] = one(T) / η_val
            shared_mem[end-1] = -shifti[1] + val_ζ * inv_1_plus_w1  # c for Δs
        end
        sync_threads()
        
        ζ_s = shared_mem[end-3]
        inv_η = shared_mem[end-2]
        c_s = shared_mem[end-1]
        
        # 更新 step_s
        idx = tid
        while idx <= size_i
            if idx == 1
                step_si[idx] = inv_η * (wi[1] * shifti[1] - ζ_s)
            else
                step_si[idx] = inv_η * (shifti[idx] + c_s * wi[idx])
            end
            idx += blockDim().x
        end
        sync_threads()
        
        # Step 3: 计算 shift = W⁻¹Δs ∘ WΔz - σμe
        # 先计算内积
        val_dot = zero(T)
        idx = tid
        while idx <= size_i
            val_dot += step_si[idx] * step_zi[idx]
            idx += blockDim().x
        end
        val_dot = block_reduce_sum(val_dot, shared_mem)
        
        if tid == 1
            shared_mem[end] = val_dot
            shared_mem[end-7] = step_si[1]  # s0
            shared_mem[end-6] = step_zi[1]  # z0
        end
        sync_threads()
        
        val_dot = shared_mem[end]
        s0 = shared_mem[end-7]
        z0 = shared_mem[end-6]
        
        # 最终更新shift
        idx = tid
        while idx <= size_i
            if idx == 1
                shifti[idx] = val_dot - σμ
            else
                shifti[idx] = s0 * step_zi[idx] + z0 * step_si[idx]
            end
            idx += blockDim().x
        end
    end
    
    return nothing
end

# 单个大型cone的特殊kernel（使用单块避免网格同步）
function _kernel_combined_ds_shift_soc_single_cone!(
    shift::AbstractVector{T},
    step_z::AbstractVector{T},
    step_s::AbstractVector{T},
    w::AbstractVector{T},
    η::T,
    rng_cone,
    cone_size::Cint,
    σμ::T
) where {T}
    tid = Cint(threadIdx().x)
    
    # 共享内存
    shared_size = cld(blockDim().x, 32) + 10
    shared_mem = @cuDynamicSharedMem(T, shared_size)
    
    # Step 1: 复制step_z到shift作为临时空间
    idx = tid
    while idx <= cone_size
        @inbounds shift[rng_cone.start + idx - 1] = step_z[rng_cone.start + idx - 1]
        idx += blockDim().x
    end
    sync_threads()
    
    # 计算 ζ_z = Σ(w[j]*tmp[j]) for j=2:end
    local_ζ = zero(T)
    idx = tid
    while idx <= cone_size - 1  # 从索引2开始
        @inbounds local_ζ += w[rng_cone.start + idx] * shift[rng_cone.start + idx]
        idx += blockDim().x
    end
    
    local_ζ = block_reduce_sum(local_ζ, shared_mem)
    
    if tid == 1
        shared_mem[end-9] = local_ζ  # ζ_z
    end
    sync_threads()
    
    ζ_z = shared_mem[end-9]
    
    # 计算系数
    if tid == 1
        @inbounds w1_val = w[rng_cone.start]
        @inbounds tmp1_val = shift[rng_cone.start]
        inv_1_plus_w1 = one(T) / (one(T) + w1_val)
        c_z = tmp1_val + ζ_z * inv_1_plus_w1
        shared_mem[end-6] = inv_1_plus_w1
        shared_mem[end-5] = c_z
        shared_mem[end-4] = η
        shared_mem[end-3] = one(T) / η
        shared_mem[end-2] = w1_val * tmp1_val + ζ_z  # 用于step_z[1]
    end
    sync_threads()
    
    inv_1_plus_w1 = shared_mem[end-6]
    c_z = shared_mem[end-5]
    η_val = shared_mem[end-4]
    inv_η = shared_mem[end-3]
    wz1_val = shared_mem[end-2]
    
    # 更新 step_z
    idx = tid
    while idx <= cone_size
        if idx == 1
            @inbounds step_z[rng_cone.start] = η_val * wz1_val
        else
            @inbounds tmp_val = shift[rng_cone.start + idx - 1]
            @inbounds w_val = w[rng_cone.start + idx - 1]
            @inbounds step_z[rng_cone.start + idx - 1] = η_val * (tmp_val + c_z * w_val)
        end
        idx += blockDim().x
    end
    sync_threads()
    
    # Step 2: 复制step_s到shift作为临时空间
    idx = tid
    while idx <= cone_size
        @inbounds shift[rng_cone.start + idx - 1] = step_s[rng_cone.start + idx - 1]
        idx += blockDim().x
    end
    sync_threads()
    
    # 计算 ζ_s
    local_ζ = zero(T)
    idx = tid
    while idx <= cone_size - 1  # 从索引2开始
        @inbounds local_ζ += w[rng_cone.start + idx] * shift[rng_cone.start + idx]
        idx += blockDim().x
    end
    
    local_ζ = block_reduce_sum(local_ζ, shared_mem)
    
    if tid == 1
        shared_mem[end-8] = local_ζ  # ζ_s
    end
    sync_threads()
    
    ζ_s = shared_mem[end-8]
    
    # 计算系数
    if tid == 1
        @inbounds tmp1_val = shift[rng_cone.start]
        @inbounds w1_val = w[rng_cone.start]
        c_s = -tmp1_val + ζ_s * inv_1_plus_w1
        shared_mem[end-1] = c_s
        shared_mem[end] = w1_val * tmp1_val - ζ_s  # 用于step_s[1]
    end
    sync_threads()
    
    c_s = shared_mem[end-1]
    ws1_val = shared_mem[end]
    
    # 更新 step_s
    idx = tid
    while idx <= cone_size
        if idx == 1
            @inbounds step_s[rng_cone.start] = inv_η * ws1_val
        else
            @inbounds tmp_val = shift[rng_cone.start + idx - 1]
            @inbounds w_val = w[rng_cone.start + idx - 1]
            @inbounds step_s[rng_cone.start + idx - 1] = inv_η * (tmp_val + c_s * w_val)
        end
        idx += blockDim().x
    end
    sync_threads()
    
    # Step 3: 计算内积
    local_dot = zero(T)
    idx = tid
    while idx <= cone_size
        @inbounds local_dot += step_s[rng_cone.start + idx - 1] * step_z[rng_cone.start + idx - 1]
        idx += blockDim().x
    end
    
    local_dot = block_reduce_sum(local_dot, shared_mem)
    
    if tid == 1
        shared_mem[end-7] = local_dot  # val_dot
    end
    sync_threads()
    
    val_dot = shared_mem[end-7]
    
    # 获取s0和z0
    if tid == 1
        @inbounds shared_mem[end-9] = step_s[rng_cone.start]  # s0
        @inbounds shared_mem[end-8] = step_z[rng_cone.start]  # z0
    end
    sync_threads()
    
    s0 = shared_mem[end-9]
    z0 = shared_mem[end-8]
    
    # 最终更新shift
    idx = tid
    while idx <= cone_size
        if idx == 1
            @inbounds shift[rng_cone.start] = val_dot - σμ
        else
            @inbounds sz_val = step_z[rng_cone.start + idx - 1]
            @inbounds ss_val = step_s[rng_cone.start + idx - 1]
            @inbounds shift[rng_cone.start + idx - 1] = s0 * sz_val + z0 * ss_val
        end
        idx += blockDim().x
    end
    
    return nothing
end

function _kernel_combined_ds_shift_soc!(
    shift::AbstractVector{T},
    step_z::AbstractVector{T},
    step_s::AbstractVector{T},
    w::AbstractVector{T},
    η::AbstractVector{T},
    rng_cones::AbstractVector,
    n_linear::Cint,
    n_soc::Cint,
    σμ::T
) where {T}

    i = (blockIdx().x-1)*blockDim().x+threadIdx().x

    if i <= n_soc
        shift_i = i + n_linear
        rng_cone_i = rng_cones[shift_i]
        size_i = length(rng_cone_i)
        @views step_zi = step_z[rng_cone_i] 
        @views step_si = step_s[rng_cone_i] 
        @views wi = w[rng_cone_i] 
        @views shifti = shift[rng_cone_i] 
    
        #shift vector used as workspace for a few steps 
        tmp = shifti            

        #Δz <- WΔz
        @inbounds for j in 1:size_i
            tmp[j] = step_zi[j]
        end         
        ζ = zero(T)
        
        @inbounds for j in 2:size_i
            ζ += wi[j]*tmp[j]
        end

        c = tmp[1] + ζ/(1+wi[1])
      
        step_zi[1] = η[i]*(wi[1]*tmp[1] + ζ)
      
        @inbounds for j in 2:size_i
            step_zi[j] = η[i]*(tmp[j] + c*wi[j]) 
        end      

        #Δs <- W⁻¹Δs
        @inbounds for j in 1:size_i
            tmp[j] = step_si[j]
        end           
        ζ = zero(T)
        @inbounds for j in 2:size_i
            ζ += wi[j]*tmp[j]
        end

        c = -tmp[1] + ζ/(1+wi[1])
    
        step_si[1] = (one(T)/η[i])*(wi[1]*tmp[1] - ζ)
    
        @inbounds for j = 2:size_i
            step_si[j] = (one(T)/η[i])*(tmp[j] + c*wi[j])
        end

        #shift = W⁻¹Δs ∘ WΔz - σμe  
        val = zero(T)
        @inbounds for j in 1:size_i
            val += step_si[j]*step_zi[j]
        end       
        shifti[1] = val - σμ 

        s0   = step_si[1]
        z0   = step_zi[1]
        for j = 2:size_i
            shifti[j] = s0*step_zi[j] + z0*step_si[j]
        end      
    end                    

    return nothing
end

@inline function combined_ds_shift_soc!(
    shift::AbstractVector{T},
    step_z::AbstractVector{T},
    step_s::AbstractVector{T},
    w::AbstractVector{T},
    η::AbstractVector{T},
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_soc::Cint,
    σμ::T
) where {T}
    if n_soc == 1
        # 特殊优化：单个大型cone
        CUDA.@allowscalar begin
            rng_cone = rng_cones[n_shift + 1]
            cone_size = Cint(length(rng_cone))
            η_val = η[1]
        end
        
        kernel = @cuda launch=false _kernel_combined_ds_shift_soc_single_cone!(shift, step_z, step_s, w, η_val, rng_cone, cone_size, σμ)
        config = launch_configuration(kernel.fun)
        threads = min(1024, config.threads)  # 使用更多线程处理大型cone
        blocks = 1  # 单块执行
        
        shmem_size = (cld(threads, 32) + 10) * sizeof(T)
        CUDA.@sync kernel(shift, step_z, step_s, w, η_val, rng_cone, cone_size, σμ; threads, blocks, shmem=shmem_size)
    elseif should_use_optimized_version(rng_cones, n_shift, n_soc)
        # 使用优化版本（少量大型cone）
        config = get_optimized_config(rng_cones, n_shift, n_soc)
        shared_mem_size = sizeof(T) * (config.threads ÷ 32 + 8)
        
        kernel = @cuda launch=false _kernel_combined_ds_shift_soc_optimized!(shift, step_z, step_s, w, η, rng_cones, n_shift, n_soc, σμ)
        CUDA.@sync kernel(shift, step_z, step_s, w, η, rng_cones, n_shift, n_soc, σμ; 
                         threads=config.threads, blocks=n_soc, shmem=shared_mem_size)
    else
        # 原始实现：多个cone
        kernel = @cuda launch=false _kernel_combined_ds_shift_soc!(shift, step_z, step_s, w, η, rng_cones, n_shift, n_soc, σμ)
        config = launch_configuration(kernel.fun)
        threads = min(n_soc, config.threads)
        blocks = cld(n_soc, threads)

        CUDA.@sync kernel(shift, step_z, step_s, w, η, rng_cones, n_shift, n_soc, σμ; threads, blocks)
    end
end

function _kernel_Δs_from_Δz_offset_soc!(
    out::AbstractVector{T},
    ds::AbstractVector{T},
    z::AbstractVector{T},
    w::AbstractVector{T},
    λ::AbstractVector{T},
    η::AbstractVector{T},
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_soc::Cint
) where {T}

    i = (blockIdx().x-1)*blockDim().x+threadIdx().x

    if i <= n_soc
        shift_i = i + n_shift
        rng_cone_i = rng_cones[shift_i]
        size_i = length(rng_cone_i)
        @views outi = out[rng_cone_i] 
        @views dsi = ds[rng_cone_i] 
        @views zi = z[rng_cone_i] 
        @views wi = w[rng_cone_i] 
        @views λi = λ[rng_cone_i] 

        #out = Wᵀ(λ \ ds).  Below is equivalent,
        #but appears to be a little more stable 
        reszi = _soc_residual_gpu(zi)

        @views λ1ds1  = _dot_xy_gpu(λi,dsi,2:size_i)
        @views w1ds1  = _dot_xy_gpu(wi,dsi,2:size_i)

        _minus_vec_gpu(outi,zi)
        outi[1] = +zi[1]
    
        c = λi[1]*dsi[1] - λ1ds1
        _multiply_gpu(outi,c/reszi)

        outi[1] += η[i]*w1ds1
        @inbounds for j in 2:size_i
            outi[j] += η[i]*(dsi[j] + w1ds1/(1+wi[1])*wi[j])
        end
    
        _multiply_gpu(outi,one(T)/λi[1])
    end

    return nothing

end

# 优化版本：块内并行处理每个cone
function _kernel_Δs_from_Δz_offset_soc_optimized!(
    out::AbstractVector{T},
    ds::AbstractVector{T},
    z::AbstractVector{T},
    w::AbstractVector{T},
    λ::AbstractVector{T},
    η::AbstractVector{T},
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_soc::Cint
) where {T}
    cone_idx = blockIdx().x
    tid = threadIdx().x
    
    # 共享内存分配：用于存储中间结果
    shared_size = blockDim().x ÷ 32 + 5  # 额外空间存储reszi, λ1ds1, w1ds1, c, 1/λi[1]
    shared_mem = @cuDynamicSharedMem(T, shared_size)
    
    if cone_idx <= n_soc
        shift_i = cone_idx + n_shift
        rng_cone_i = rng_cones[shift_i]
        size_i = length(rng_cone_i)
        @views outi = out[rng_cone_i]
        @views dsi = ds[rng_cone_i]
        @views zi = z[rng_cone_i]
        @views wi = w[rng_cone_i]
        @views λi = λ[rng_cone_i]
        
        # 并行计算z的residual
        val_reszi = zero(T)
        idx = tid
        while idx <= size_i
            if idx == 1
                val_reszi += zi[idx] * zi[idx]
            else
                val_reszi -= zi[idx] * zi[idx]
            end
            idx += blockDim().x
        end
        val_reszi = block_reduce_sum(val_reszi, shared_mem)
        
        if tid == 1
            shared_mem[end-4] = val_reszi  # reszi
        end
        sync_threads()
        reszi = shared_mem[end-4]
        
        # 并行计算λ·ds点积（从索引2开始）
        val_λ1ds1 = zero(T)
        idx = 2 + (tid - 1)
        while idx <= size_i
            val_λ1ds1 += λi[idx] * dsi[idx]
            idx += blockDim().x
        end
        val_λ1ds1 = block_reduce_sum(val_λ1ds1, shared_mem)
        
        if tid == 1
            shared_mem[end-3] = val_λ1ds1  # λ1ds1
        end
        sync_threads()
        λ1ds1 = shared_mem[end-3]
        
        # 并行计算w·ds点积（从索引2开始）
        val_w1ds1 = zero(T)
        idx = 2 + (tid - 1)
        while idx <= size_i
            val_w1ds1 += wi[idx] * dsi[idx]
            idx += blockDim().x
        end
        val_w1ds1 = block_reduce_sum(val_w1ds1, shared_mem)
        
        if tid == 1
            shared_mem[end-2] = val_w1ds1  # w1ds1
            shared_mem[end-1] = (λi[1] * dsi[1] - λ1ds1) / reszi  # c
            shared_mem[end] = one(T) / λi[1]  # 1/λi[1]
        end
        sync_threads()
        
        w1ds1 = shared_mem[end-2]
        c = shared_mem[end-1]
        inv_λi1 = shared_mem[end]
        
        # 并行计算输出向量
        idx = tid
        while idx <= size_i
            if idx == 1
                outi[idx] = (zi[idx] * c + η[cone_idx] * w1ds1) * inv_λi1
            else
                outi[idx] = (-zi[idx] * c + η[cone_idx] * (dsi[idx] + w1ds1/(1+wi[1])*wi[idx])) * inv_λi1
            end
            idx += blockDim().x
        end
    end
    
    return nothing
end

# 单个大型cone的特殊kernel（使用单块避免网格同步）
function _kernel_Δs_from_Δz_offset_soc_single_cone!(
    out::AbstractVector{T},
    ds::AbstractVector{T},
    z::AbstractVector{T},
    w::AbstractVector{T},
    λ::AbstractVector{T},
    η::T,
    rng_cone,
    cone_size::Cint
) where {T}
    tid = Cint(threadIdx().x)
    
    # 共享内存用于规约和存储结果
    shared_size = cld(blockDim().x, 32) + 10
    shared_mem = @cuDynamicSharedMem(T, shared_size)
    
    # Step 1: 计算z的residual
    local_reszi = zero(T)
    idx = tid
    while idx <= cone_size
        @inbounds val = z[rng_cone.start + idx - 1]
        if idx == 1
            local_reszi += val * val
        else
            local_reszi -= val * val
        end
        idx += blockDim().x
    end
    
    # 块内规约
    local_reszi = block_reduce_sum(local_reszi, shared_mem)
    
    if tid == 1
        shared_mem[end-4] = local_reszi  # reszi
    end
    sync_threads()
    
    reszi = shared_mem[end-4]
    
    # Step 2: 计算λ·ds点积（从索引2开始）
    local_λ1ds1 = zero(T)
    idx = tid
    while idx <= cone_size - 1  # 从索引2开始
        @inbounds local_λ1ds1 += λ[rng_cone.start + idx] * ds[rng_cone.start + idx]
        idx += blockDim().x
    end
    
    local_λ1ds1 = block_reduce_sum(local_λ1ds1, shared_mem)
    
    if tid == 1
        shared_mem[end-3] = local_λ1ds1  # λ1ds1
    end
    sync_threads()
    
    λ1ds1 = shared_mem[end-3]
    
    # Step 3: 计算w·ds点积（从索引2开始）
    local_w1ds1 = zero(T)
    idx = tid
    while idx <= cone_size - 1  # 从索引2开始
        @inbounds local_w1ds1 += w[rng_cone.start + idx] * ds[rng_cone.start + idx]
        idx += blockDim().x
    end
    
    local_w1ds1 = block_reduce_sum(local_w1ds1, shared_mem)
    
    if tid == 1
        shared_mem[end-2] = local_w1ds1  # w1ds1
    end
    sync_threads()
    
    w1ds1 = shared_mem[end-2]
    
    # 计算系数
    if tid == 1
        @inbounds λ1_val = λ[rng_cone.start]
        @inbounds ds1_val = ds[rng_cone.start]
        @inbounds w1_val = w[rng_cone.start]
        c = (λ1_val * ds1_val - λ1ds1) / reszi
        inv_λ1 = one(T) / λ1_val
        shared_mem[end-1] = c
        shared_mem[end] = inv_λ1
        shared_mem[end-5] = one(T) / (one(T) + w1_val)  # 1/(1+w[1])
    end
    sync_threads()
    
    c = shared_mem[end-1]
    inv_λ1 = shared_mem[end]
    inv_1_plus_w1 = shared_mem[end-5]
    
    # Step 4: 并行计算输出向量
    idx = tid
    while idx <= cone_size
        if idx == 1
            @inbounds z1_val = z[rng_cone.start]
            @inbounds out[rng_cone.start] = (z1_val * c + η * w1ds1) * inv_λ1
        else
            @inbounds zi_val = z[rng_cone.start + idx - 1]
            @inbounds dsi_val = ds[rng_cone.start + idx - 1]
            @inbounds wi_val = w[rng_cone.start + idx - 1]
            @inbounds out[rng_cone.start + idx - 1] = (-zi_val * c + η * (dsi_val + w1ds1 * inv_1_plus_w1 * wi_val)) * inv_λ1
        end
        idx += blockDim().x
    end
    
    return nothing
end

@inline function Δs_from_Δz_offset_soc!(
    out::AbstractVector{T},
    ds::AbstractVector{T},
    z::AbstractVector{T},
    w::AbstractVector{T},
    λ::AbstractVector{T},
    η::AbstractVector{T},
    rng_cones::AbstractVector,
    n_shift::Cint,
    n_soc::Cint
) where {T}
    if n_soc == 1
        # 特殊优化：单个大型cone
        CUDA.@allowscalar begin
            rng_cone = rng_cones[n_shift + 1]
            cone_size = Cint(length(rng_cone))
            η_val = η[1]
        end
        
        kernel = @cuda launch=false _kernel_Δs_from_Δz_offset_soc_single_cone!(out, ds, z, w, λ, η_val, rng_cone, cone_size)
        config = launch_configuration(kernel.fun)
        threads = min(1024, config.threads)  # 使用更多线程处理大型cone
        blocks = 1  # 单块执行
        
        shmem_size = (cld(threads, 32) + 10) * sizeof(T)
        CUDA.@sync kernel(out, ds, z, w, λ, η_val, rng_cone, cone_size; threads, blocks, shmem=shmem_size)
    elseif should_use_optimized_version(rng_cones, n_shift, n_soc)
        # 使用优化版本（少量大型cone）
        config = get_optimized_config(rng_cones, n_shift, n_soc)
        shared_mem_size = sizeof(T) * (config.threads ÷ 32 + 5)
        
        kernel = @cuda launch=false _kernel_Δs_from_Δz_offset_soc_optimized!(out, ds, z, w, λ, η, rng_cones, n_shift, n_soc)
        CUDA.@sync kernel(out, ds, z, w, λ, η, rng_cones, n_shift, n_soc; 
                         threads=config.threads, blocks=n_soc, shmem=shared_mem_size)
    else
        # 使用原始版本
        kernel = @cuda launch=false _kernel_Δs_from_Δz_offset_soc!(out, ds, z, w, λ, η, rng_cones, n_shift, n_soc)
        config = launch_configuration(kernel.fun)
        threads = min(n_soc, config.threads)
        blocks = cld(n_soc, threads)

        CUDA.@sync kernel(out, ds, z, w, λ, η, rng_cones, n_shift, n_soc; threads, blocks)
    end
end

#return maximum allowable step length while remaining in the socone
function _kernel_step_length_soc(
    dz::AbstractVector{T},
    ds::AbstractVector{T},
     z::AbstractVector{T},
     s::AbstractVector{T},
     α::AbstractVector{T},
     rng_cones::AbstractVector,
     n_linear::Cint,
     n_soc::Cint
) where {T}

    i = (blockIdx().x-1)*blockDim().x+threadIdx().x

    if i <= n_soc
        shift_i = i + n_linear
        rng_cone_i = rng_cones[shift_i]
        @views si = s[rng_cone_i] 
        @views dsi = ds[rng_cone_i] 
        @views zi = z[rng_cone_i] 
        @views dzi = dz[rng_cone_i]         

        αz   = _step_length_soc_component_gpu(zi,dzi,α[i])
        αs   = _step_length_soc_component_gpu(si,dsi,α[i])
        α[i] = min(αz,αs)
    end

    return nothing
end

@inline function step_length_soc(
    dz::AbstractVector{T},
    ds::AbstractVector{T},
     z::AbstractVector{T},
     s::AbstractVector{T},
     α::AbstractVector{T},
     αmax::T,
     rng_cones::AbstractVector,
     n_shift::Cint,
     n_soc::Cint
) where {T}

    kernel = @cuda launch=false _kernel_step_length_soc(dz, ds, z, s, α, rng_cones, n_shift, n_soc)
    config = launch_configuration(kernel.fun)
    threads = min(n_soc, config.threads)
    blocks = cld(n_soc, threads)

    CUDA.@sync kernel(dz, ds, z, s, α, rng_cones, n_shift, n_soc; threads, blocks)
    @views αmax = min(αmax,minimum(α[1:n_soc]))

    return αmax
end

function _kernel_compute_barrier_soc(
    barrier::AbstractVector{T},
    z::AbstractVector{T},
    s::AbstractVector{T},
    dz::AbstractVector{T},
    ds::AbstractVector{T},
    α::T,
    rng_cones::AbstractVector,
    n_linear::Cint,
    n_soc::Cint
) where {T}

    i = (blockIdx().x-1)*blockDim().x+threadIdx().x

    if i <= n_soc
        shift_i = i + n_linear
        rng_cone_i = rng_cones[shift_i]
        @views si = s[rng_cone_i] 
        @views dsi = ds[rng_cone_i] 
        @views zi = z[rng_cone_i] 
        @views dzi = dz[rng_cone_i]  
        res_si = _soc_residual_shifted(si,dsi,α)
        res_zi = _soc_residual_shifted(zi,dzi,α)

        # avoid numerical issue if res_s <= 0 or res_z <= 0
        barrier[i] = (res_si > 0 && res_zi > 0) ? -logsafe(res_si*res_zi)/2 : Inf
    end

    return nothing
end

@inline function compute_barrier_soc(
    barrier::AbstractVector{T},
    z::AbstractVector{T},
    s::AbstractVector{T},
    dz::AbstractVector{T},
    ds::AbstractVector{T},
    α::T,
    rng_cones::AbstractVector,
    n_linear::Cint,
    n_soc::Cint
) where {T}

    kernel = @cuda launch=false _kernel_compute_barrier_soc(barrier,z,s,dz,ds,α,rng_cones,n_linear,n_soc)
    config = launch_configuration(kernel.fun)
    threads = min(n_soc, config.threads)
    blocks = cld(n_soc, threads)

    CUDA.@sync kernel(barrier,z,s,dz,ds,α,rng_cones,n_linear,n_soc; threads, blocks)

    return sum(barrier[1:n_soc])
end

# # ---------------------------------------------
# # operations supported by symmetric cones only 
# # ---------------------------------------------


# # implements y = αWx + βy for the socone
# function mul_W!(
#     K::SecondOrderCone{T},
#     is_transpose::Symbol,
#     y::AbstractVector{T},
#     x::AbstractVector{T},
#     α::T,
#     β::T
# ) where {T}

#   #NB: symmetric, so ignore transpose

#   # use the fast product method from ECOS ECC paper
#   @views ζ = dot(K.w[2:end],x[2:end])
#   c = x[1] + ζ/(1+K.w[1])

#   y[1] = α*K.η*(K.w[1]*x[1] + ζ) + β*y[1]

#   @inbounds for i in 2:length(y)
#       y[i] = α*K.η*(x[i] + c*K.w[i]) + β*y[i]
#   end

#   return nothing
# end

# # implements y = αW^{-1}x + βy for the socone
# function mul_Winv!(
#     K::SecondOrderCone{T},
#     is_transpose::Symbol,
#     y::AbstractVector{T},
#     x::AbstractVector{T},
#     α::T,
#     β::T
# ) where {T}

#     #NB: symmetric, so ignore transpose

#     # use the fast inverse product method from ECOS ECC paper
#     @views ζ = dot(K.w[2:end],x[2:end])
#     c = -x[1] + ζ/(1+K.w[1])

#     y[1] = (α/K.η)*(K.w[1]*x[1] - ζ) + β*y[1]

#     @inbounds for i = 2:length(y)
#         y[i] = (α/K.η)*(x[i] + c*K.w[i]) + β*y[i]
#     end

#     return nothing
# end

# # implements x = λ \ z for the socone, where λ
# # is the internally maintained scaling variable.
# function λ_inv_circ_op!(
#     K::SecondOrderCone{T},
#     x::AbstractVector{T},
#     z::AbstractVector{T}
# ) where {T}

#     inv_circ_op!(K, x, K.λ, z)

# end

# ---------------------------------------------
# Jordan algebra operations for symmetric cones 
# ---------------------------------------------

# # implements x = y \ z for the socone
# function inv_circ_op!(
#     K::SecondOrderCone{T},
#     x::AbstractVector{T},
#     y::AbstractVector{T},
#     z::AbstractVector{T}
# ) where {T}

#     p = _soc_residual(y)
#     pinv = 1/p
#     @views v = dot(y[2:end],z[2:end])

#     x[1]      = (y[1]*z[1] - v)*pinv
#     @views x[2:end] .= pinv.*(v/y[1] - z[1]).*y[2:end] + (1/y[1]).*z[2:end]

#     return nothing
# end

# ---------------------------------------------
# internal operations for second order cones 
# ---------------------------------------------

@inline function _soc_residual_gpu(z::AbstractVector{T}) where {T} 
    res = z[1]*z[1]
    @inbounds for j in 2:length(z)
        res -= z[j]*z[j]
    end
    
    return res
end 

@inline function _sqrt_soc_residual_gpu(z::AbstractVector{T}) where {T} 
    res = _soc_residual_gpu(z)
    
    # set res to 0 when z is not an interior point
    res = res > 0.0 ? sqrt(res) : zero(T)
end 

# compute the residual at z + α*dz without storing the intermediate vector
@inline function _soc_residual_shifted(
    z::AbstractVector{T}, 
    dz::AbstractVector{T}, 
    α::T
) where {T} 
    
    x0 = z[1] + α * dz[1]
    # compute dot product of shifted vector
    x1sq = zero(T)
    @inbounds for j in 2:length(z)
        xj = z[j] + α * dz[j]
        x1sq += xj * xj
    end
    x1norm = sqrt(x1sq)
    res = (x0 - x1norm) * (x0 + x1norm)
    return res
end 

@inline function logsafe(v::T) where {T<:Real}
    if v < 0
        return -typemax(T)
    else 
        return log(v)
    end
end

@inline function _dot_xy_gpu(x::AbstractVector{T},y::AbstractVector{T},rng::UnitRange) where {T} 
    val = zero(T)
    @inbounds for j in rng
        val += x[j]*y[j]
    end
    
    return val
end 

@inline function _minus_vec_gpu(y::AbstractVector{T},x::AbstractVector{T}) where {T} 
    @inbounds for j in 1:length(x)
        y[j] = -x[j]
    end
end 

@inline function _multiply_gpu(x::AbstractVector{T},a::T) where {T} 
    @inbounds for j in 1:length(x)
        x[j] *= a 
    end
end 

# find the maximum step length α≥0 so that
# x + αy stays in the SOC
@inline function _step_length_soc_component_gpu(
    x::AbstractVector{T},
    y::AbstractVector{T},
    αmax::T
) where {T}

    if x[1] >= 0 && y[1] < 0
        αmax = min(αmax,-x[1]/y[1])
    end

    # assume that x is in the SOC, and find the minimum positive root
    # of the quadratic equation:  ||x₁+αy₁||^2 = (x₀ + αy₀)^2

    @views a = _soc_residual_gpu(y) #NB: could be negative
    @views b = 2*(x[1]*y[1] - _dot_xy_gpu(x,y,2:length(x)))
    @views c = max(zero(T),_soc_residual_gpu(x)) #should be ≥0
    d = b^2 - 4*a*c

    if(c < 0)
        # This should never be reachable since c ≥ 0 above
        return -Inf
    end

    if( (a > 0 && b > 0) || d < 0)
        #all negative roots / complex root pair
        #-> infinite step length
        return αmax

    elseif a == 0
        #edge case with only one root.  This corresponds to
        #the case where the search direction is exactly on the 
        #cone boundary.   The root should be -c/b, but b can't 
        #be negative since both (x,y) are in the cone and it is 
        #self dual, so <x,y> \ge 0 necessarily.
        return αmax

    elseif c == 0
        #Edge case with one of the roots at 0.   This corresponds 
        #to the case where the initial point is exactly on the 
        #cone boundary.  The other root is -b/a.   If the search 
        #direction is in the cone, then a >= 0 and b can't be 
        #negative due to self-duality.  If a < 0, then the 
        #direction is outside the cone and b can't be positive.
        #Either way, step length is determined by whether or not 
        #the search direction is in the cone.

        return (a >= 0 ? αmax : zero(T)) 
    end 


    # if we got this far then we need to calculate a pair 
    # of real roots and choose the smallest positive one.  
    # We need to be cautious about cancellations though.  
    # See §1.4: Goldberg, ACM Computing Surveys, 1991 
    # https://dl.acm.org/doi/pdf/10.1145/103162.103163

    t = (b >= 0) ? (-b - sqrt(d)) : (-b + sqrt(d))

    r1 = (2*c)/t;
    r2 = t/(2*a);

    #return the minimum positive root, up to αmax
    r1 = r1 < 0 ? floatmax(T) : r1
    r2 = r2 < 0 ? floatmax(T) : r2

    return min(αmax,r1,r2)

end

# End of SOC GPU operations
