export AbstractPreprocessor,
       Chain,
       Preprocessor,
       FourierPreprocessor,
       PolynomialPreprocessor,
       TilingPreprocessor,
       RadialBasisFunctions,
       RandomProjection,
       SparseRandomProjection,
       ImageResize,
       ImageResizeNearestNeighbour,
       ImageResizeBilinear,
       ImageCrop,
       StackFrames

using .Utils: Tiling, encode
using LinearAlgebra: norm
using Flux: Chain
using ImageTransformations: imresize!
using ShiftedArrays:CircShiftedVector

"""
Preprocess an [`Observation`](@ref) and return a new observation.
By default, the preprocessor is only applied to the state field
of the [`Observation`](@ref) and other fields remain unchanged.

For customized preprocessors that inherit `AbstractPreprocessor`,
you can change this behavior by rewriting
`(p::AbstractPreprocessor)(obs::Observation)` method.
"""
abstract type AbstractPreprocessor end

(p::AbstractPreprocessor)(obs::Observation) =
    Observation(
        obs.reward,
        obs.terminal,
        p(obs.state),
        merge(obs.meta, (; Symbol(:state_before_, typeof(p).name) => obs.state)),
    )

"""
    FourierPreprocessor(order::Int)

Transform a scalar to a vector of `order+1` Fourier bases.
"""
struct FourierPreprocessor <: AbstractPreprocessor
    order::Int
end

(p::FourierPreprocessor)(s::Number) = [cos(i * π * s) for i = 0:p.order]

"""
    PolynomialPreprocessor(order::Int)

Transform a scalar to vector of maximum `order` polynomial.
"""
struct PolynomialPreprocessor <: AbstractPreprocessor
    order::Int
end

(p::PolynomialPreprocessor)(s::Number) = [s^i for i = 0:p.order]

"""
    TilingPreprocessor(tilings::Vector{<:Tiling})

Use each `tilings` to encode the state and return a vector.
"""
struct TilingPreprocessor{Tt<:Tiling} <: AbstractPreprocessor
    tilings::Vector{Tt}
end

(p::TilingPreprocessor)(s::Union{<:Number,<:Array}) = [encode(t, s) for t in p.tilings]

"""
    ImageResize(img::Array{T, N})
    ImageResize(dims::Int...) -> ImageResize(Float32, dims...)
    ImageResize(T::Type{<:Number}, dims::Int...)

Using BSpline method to resize the `state` field of an [`Observation`](@ref) to size of `img` (or `dims`).
"""
struct ImageResize{T, N} <: AbstractPreprocessor
    img::Array{T, N}
end

ImageResize(dims::Int...) = ImageResize(Float32, dims...)
ImageResize(T::Type{<:Number}, dims::Int...) = ImageResize(Array{T}(undef, dims))

function (p::ImageResize)(obs::Observation)
    imresize!(p.img, obs.state)
    Observation(obs.reward, obs.terminal, p.img, obs.meta)
end

"""
    struct ImageCrop
        xidx::UnitRange{Int64}
        yidx::UnitRange{Int64}
    end

Select indices `xidx` and `yidx` from a 2 or 3 dimensional array.
TODO: Inefficient!

# Example:

```
c = ImageCrop(2:5, 3:2:9)
c([10i + j for i in 1:10, j in 1:10])
```
"""
struct ImageCrop{Tx,Ty} <: AbstractPreprocessor
    xidx::Tx
    yidx::Ty
end

(c::ImageCrop)(x::Array{T,2}) where {T} = x[c.xidx, c.yidx]
(c::ImageCrop)(x::Array{T,3}) where {T} = x[:, c.xidx, c.yidx]


"""
    struct ImageResizeNearestNeighbour
        outdim::Tuple{Int64, Int64}
    end

Resize any image to `outdim = (width, height)` by nearest-neighbour
interpolation (i.e. subsampling).

# Example:

```
r = ImageResizeNearestNeighbour((50, 50))
r(rand(200, 200))
r(rand(UInt8, 3, 100, 100))
```
"""
struct ImageResizeNearestNeighbour <: AbstractPreprocessor
    outdim::Tuple{Int64,Int64}
end
function (r::ImageResizeNearestNeighbour)(x)
    indim = size(x)
    xidx = round.(Int64, collect(1:r.outdim[1]) .* indim[end-1] / r.outdim[1])
    yidx = round.(Int64, collect(1:r.outdim[2]) .* indim[end] / r.outdim[2])
    length(indim) > 2 ? x[:, xidx, yidx] : x[xidx, yidx]
end

struct SparseRandomProjection <: AbstractPreprocessor
    w::Array{Float64,2}
    b::Array{Float64,1}
end

(p::SparseRandomProjection)(s) = clamp.(p.w * s + p.b, 0, Inf)

struct RandomProjection <: AbstractPreprocessor
    w::Array{Float64,2}
end
(p::RandomProjection)(s) = p.w * s

struct RadialBasisFunctions <: AbstractPreprocessor
    means::Array{Array{Float64,1},1}
    sigmas::Array{Float64,1}
    state::Array{Float64,1}
end

struct Box{T}
    low::Array{T,1}
    high::Array{T,1}
end

function RadialBasisFunctions(box::Box, n, sigma)
    dim = length(box.low)
    means = [rand(dim) .* (box.high - box.low) .+ box.low for _ = 1:n]
    RadialBasisFunctions(means, typeof(sigma) <: Number ? fill(sigma, n) : sigma, zeros(n))
end

function (p::RadialBasisFunctions)(s)
    @inbounds for i = 1:length(p.state)
        p.state[i] = exp(-norm(s - p.means[i]) / p.sigmas[i])
    end
    p.state
end

"""
    StackFrames(::Type{T}=Float32, d::Int...)

Use a pre-initialized [`CircularArrayBuffer`](@ref) to store the latest several states specified by `d`.
Before processing any observation, the buffer is filled with `zero{T}`.
"""
mutable struct StackFrames{T, N} <: AbstractPreprocessor
    buffer::CircularArrayBuffer{T, N}
    StackFrames(d::Int...) = StackFrames(Float32, d...)
    function StackFrames(::Type{T}, d::Vararg{Int, N}) where {T, N}
        p = new{T, N}(CircularArrayBuffer{T}(d...))
        for _ in 1:capacity(p.buffer)
            push!(p.buffer, zeros(T, size(p.buffer)[1:N-1]))
        end
        p
    end
end

function (p::StackFrames{T, N})(obs::Observation) where {T, N}
    push!(p.buffer, obs.state)

    Observation(
        obs.reward,
        obs.terminal,
        p.buffer,
        obs.meta
    )
end