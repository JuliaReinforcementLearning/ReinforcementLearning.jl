using GPUArrays, CuArrays, FillArrays
using CuArrays:threadIdx, blockIdx, blockDim

#####
# Cartesian indexing of CuArray
#####

Base.checkindex(::Type{Bool}, inds::Tuple, I::CuArray{<:CartesianIndex}) = true

function Base.getindex(xs::CuArray{T, N}, indices::CuArray{CartesianIndex{N}}) where {T, N}
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

function Base.setindex!(xs::CuArray{T, N}, v::CuArray{T}, indices::CuArray{CartesianIndex{N}}) where {T, N}
    @assert length(indices) == length(v) "$xs, $(size(xs)), $v, $(size(v)), $indices, $(size(indices))"
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

function Base.setindex!(xs::CuArray{T, N}, v::T, indices::CuArray{CartesianIndex{N}}) where {T, N}
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

Base.setindex!(xs::CuArray{T, N}, v::Fill{T}, indices::CuArray{CartesianIndex{N}}) where {T, N} = setindex!(xs, v.value, indices)