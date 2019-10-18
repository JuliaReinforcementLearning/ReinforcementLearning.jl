export AbstractPreprocessor,
       Chain,
       Preprocessor,
       FourierPreprocessor,
       PolynomialPreprocessor,
       TilingPreprocessor,
       RadialBasisFunctions,
       RandomProjection,
       SparseRandomProjection,
       ImagePreprocessor,
       ImageResizeNearestNeighbour,
       ImageResizeBilinear,
       ImageCrop

using .Utils: Tiling, encode
using LinearAlgebra: norm
using Flux: Chain

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
    struct ImageCrop
        xidx::UnitRange{Int64}
        yidx::UnitRange{Int64}
    end

Select indices `xidx` and `yidx` from a 2 or 3 dimensional array.

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
    struct ImageResizeBilinear
        outdim::Tuple{Int64, Int64}
    end

Resize any image to `outdim = (width, height)` with bilinear interpolation.

# Example:

```
r = ImageResizeBilinear((50, 50))
r(rand(200, 200))
r(rand(UInt8, 3, 100, 100))
```
"""
struct ImageResizeBilinear <: AbstractPreprocessor
    outdim::Tuple{Int64,Int64}
end
for N = 2:3
    @eval @__MODULE__() function (p::ImageResizeBilinear)(x::Array{T,$N}) where {T}
        indim = size(x)
        sx, sy = (indim[end-1] - 1) / (p.outdim[1] + 1),
            (indim[end] - 1) / (p.outdim[2] + 1)
        $(N == 2 ? :(y = zeros(p.outdim[1], p.outdim[2])) :
          :(y = zeros(3, p.outdim[1], p.outdim[2])))
        for i = 1:p.outdim[1]
            for j = 1:p.outdim[2]
                r = floor(Int64, i * sx)
                c = floor(Int64, j * sy)
                dr = i * sx - r
                dc = j * sy - c
                $(N == 2 ?
                  :(y[i, j] = x[r+1, c+1] * (1 - dr) * (1 - dc) +
                              x[r+2, c+1] * dr * (1 - dc) + x[r+1, c+2] * (1 - dr) * dc +
                              x[r+2, c+2] * dr * dc) :
                  :(y[:, i, j] .= x[:, r+1, c+1] * (1 - dr) * (1 - dc) .+
                                  x[:, r+2, c+1] * dr * (1 - dc) .+
                                  x[:, r+1, c+2] * (1 - dr) * dc .+
                                  x[:, r+2, c+2] * dr * dc))
            end
        end
        y
    end
end

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

"""
    struct ImagePreprocessor
        size
        chain
    end

Use `chain` to preprocess a grayscale or color image of `size = (width, height)`.

# Example:

```
p = ImagePreprocessor((100, 100), 
                      [ImageResizeNearestNeighbour((50, 80)),
                       ImageCrop(1:30, 10:80),
                       x -> x ./ 256])
x = rand(UInt8, 100, 100)
s = ReinforcementLearning.preprocessstate(p, x)
```
"""
struct ImagePreprocessor{Ts} <: AbstractPreprocessor
    size::Ts
    chain::Array{Any,1}
end

(p::ImagePreprocessor)(s) = foldl((x, p) -> p(x), p.chain, init = reshape(s, p.size))

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