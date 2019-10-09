using GPUArrays, CuArrays, FillArrays
using CuArrays:threadIdx, blockIdx, blockDim
import Base:minimum, maximum, findmax, findmin

import Flux.Optimise: apply!, Descent, InvDecay

#####
# patch for maximum/minimum/findmax/findmin with dims==1
# considering that discrete actions in RL are usually small,
# the implementation here may be not that slow?
# watch https://github.com/JuliaGPU/CuArrays.jl/issues/304
#####
function findminmaxcol!(f, values::GPUArray{Float32, 2}, vals::GPUArray{Float32, 2}, args::GPUArray{CartesianIndex{2}, 1})
    function kernel(state, values, vals, args)
        i = linear_index(state)
        if i <= length(args)
            for j in axes(values, 1)
                @inbounds if f(values[j, i], vals[i])
                    vals[i] = values[j, i]
                    args[i] = CartesianIndex(j, i)
                end
            end
        end
        return
    end
    gpu_call(kernel, args, (values, vals, args))
    vals, args
end

function Base._findmax(values::CuArray{Float32, 2}, dims::Int)
    if dims == 1
        findminmaxcol!(Base.isgreater, values, values[1:1, :], cu(CartesianIndex.(1, axes(values, 2))))
    else
        error("unsupported yet for dims=$dims")
    end
end

function Base._findmin(values::CuArray{Float32, 2}, dims::Int)
    if dims == 1
        findminmaxcol!(Base.isless, values, values[1:1, :], cu(CartesianIndex.(1, axes(values, 2))))
    else
        error("unsupported yet for dims=$dims")
    end
end

function Base._maximum(x::CuArray{Float32, 2}, dims::Int)
    if dims == 1
        _findmax(x, dims)[1]
    else
        error("unsupported yet for dims=$dims")
    end
end

function Base._minimum(x::CuArray{Float32, 2}, dims::Int)
    if dims == 1
        _findmin(x, dims)[1]
    else
        error("unsupported yet for dims=$dims")
    end
end

#####
# Cartesian indexing of CuArray
#####

function Base.getindex(xs::CuArray{T, N}, indices::CuArray{CartesianIndex{N}, 1}) where {T, N}
  n = length(indices)
  ys = CuArray{T}(undef, n)

  if n > 0
    num_threads = min(n, 256)
    num_blocks = ceil(Int, n / num_threads)

    function kernel(ys::CuArrays.CuDeviceArray{T}, xs::CuArrays.CuDeviceArray{T}, indices)
        i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

        if i <= length(ys)
            ind = indices[i]
            ys[i] = xs[ind]
        end

        return
    end

    CuArrays.@cuda blocks=num_blocks threads=num_threads kernel(ys, xs, indices)
  end

  return ys
end

function Base.setindex!(xs::CuArray{T, N}, v::CuArray{T}, indices::CuArray{CartesianIndex{N}, 1}) where {T, N}
    # @assert length(indices) == length(v) "$xs, $(size(xs)), $v, $(size(v)), $indices, $(size(indices))"
    n = length(indices)

    if n > 0
        num_threads = min(n, 256)
        num_blocks = ceil(Int, n / num_threads)

        function kernel(xs::CuArrays.CuDeviceArray{T}, indices, v)
            i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

            if i <= length(indices)
                ind = indices[i]
                xs[ind] = v[i]
            end

            return
        end

        CuArrays.@cuda blocks=num_blocks threads=num_threads kernel(xs, indices, v)
    end
    return v
end

function Base.setindex!(xs::CuArray{T, N}, v::T, indices::CuArray{CartesianIndex{N}, 1}) where {T, N}
  n = length(indices)

  if n > 0
    num_threads = min(n, 256)
    num_blocks = ceil(Int, n / num_threads)

    function kernel(xs::CuArrays.CuDeviceArray{T}, indices, v)
        i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

        if i <= length(indices)
            ind = indices[i]
            xs[ind] = v
        end

        return
    end

    CuArrays.@cuda blocks=num_blocks threads=num_threads kernel(xs, indices, v)
  end
  return v
end

Base.setindex!(xs::CuArray{T, N}, v::Fill{T}, indices::CuArray{CartesianIndex{N}, 1}) where {T, N} = setindex!(xs, v.value, indices)