"""
    struct NoPreprocessor end
"""
struct NoPreprocessor end
export NoPreprocessor
@inline preprocessstate(p::NoPreprocessor, s) = s
@inline preprocess(::NoPreprocessor, s, r, done) = (s, r, done)
@inline preprocess(p, s, r, done) = (preprocessstate(p, s), r, done)

"""
    struct Box{T}
        low::Array{T, 1}
        high::Array{T, 1}
"""
struct Box{T}
    low::Array{T, 1}
    high::Array{T, 1}
end
"""
    struct StateAggregator
        box::Box
        ns::Int64
        nbins::Array{Int64, 1}
        offsets::Array{Int64, 1}
        perdimension::Bool
"""
struct StateAggregator
    box::Box
    ns::Int64
    nbins::Array{Int64, 1}
    offsets::Array{Int64, 1}
    perdimension::Bool
end
export StateAggregator
"""
    StateAggregator(lb::Vector, ub::Vector, nbins::Vector;
                    perdimension = false)
"""
function StateAggregator(lb::Vector, ub::Vector, nbins::Vector;
                         perdimension = false)
    if perdimension
        offsets = [0; cumsum(nbins[1:end-1])]
    else
        offsets = foldl((x, n) -> [x...; x[end] * n], nbins[1:end-1]; init = [1])
    end
    StateAggregator(Box(lb, ub), prod(nbins), nbins, offsets, perdimension)
end
"""
    StateAggregator(lb::Number, ub::Number, nbins::Int, ndims::Int; 
                    perdimension = false)
"""
StateAggregator(lb::Number, ub::Number, nbins::Int, ndims::Int; perdimension = false) =
    StateAggregator(lb * ones(ndims), ub * ones(ndims), nbins * ones(ndims))

@inline indexinbox(x, l, h, n) = round(Int64, (n - 1) * (x - l)/(h - l)) |> 
                                   i -> max(min(i, n - 1), 0)

function preprocessstate(p::StateAggregator, s)
    indices = [indexinbox(s[i], p.box.low[i], p.box.high[i], p.nbins[i]) 
               for i in 1:length(s)]
    if p.perdimension
        sp = Int64[]
        for i in 1:length(s)
            push!(sp, indices[i] + 1 + p.offsets[i])
        end
        sparsevec(sp, ones(length(s)), sum(p.nbins))
    else
        dot(indices, p.offsets) + 1
    end
end

"""
    struct TilingStateAggregator{T <: Array{StateAggregator,1}}
        ns::Int64
        tiling::T
"""
struct TilingStateAggregator{T <: Array{StateAggregator,1}}
    ns::Int64
    tiling::T
end
export TilingStateAggregator

function preprocessstate(p::TilingStateAggregator, s)
    sp = Int64[]
    istart = 0
    for tiling in p.tiling
        push!(sp, istart + preprocessstate(tiling, s))
        istart += tiling.ns
    end
    sparsevec(sp, ones(length(sp)), p.ns)
end

function tilingparams(length, nr_tiling, nr_bins_per_tile)
    w = length / (nr_bins_per_tile - (nr_tiling-1)/nr_tiling)
    offset = w * (nr_tiling-1)/nr_tiling
    k = w/nr_tiling
#     println("offset, k, w : $(offset), $(k), $(w)")
    offset, k, w
end

function TilingStateAggregator(p::StateAggregator, nr_tiling)
    length = p.box.high - p.box.low
    nr_bins_per_tile = prod(p.nbins)
    offset, k, w = tilingparams(length, nr_tiling, nr_bins_per_tile)
    stateAgg_array = [StateAggregator(p.box.low-offset+i*k,p.box.high+i*k,p.nbins)
                        for i=0:nr_tiling-1]
    TilingStateAggregator(nr_tiling * nr_bins_per_tile, stateAgg_array)
end

"""
    struct RadialBasisFunctions
        means::Array{Array{Float64, 1}, 1}
        sigmas::Array{Float64, 1}
        state::Array{Float64, 1} 
"""
struct RadialBasisFunctions
    means::Array{Array{Float64, 1}, 1}
    sigmas::Array{Float64, 1}
    state::Array{Float64, 1}
end
export RadialBasisFunctions
function RadialBasisFunctions(box::Box, n, sigma)
    dim = length(box.low)
    means = [rand(dim) .* (box.high - box.low) .+ box.low for _ in 1:n]
    RadialBasisFunctions(means, 
                         typeof(sigma) <: Number ? fill(sigma, n) : sigma, 
                         zeros(n))
end
function preprocessstate(p::RadialBasisFunctions, s)
    @inbounds for i in 1:length(p.state)
        p.state[i] = exp(-norm(s - p.means[i])/p.sigmas[i])
    end
    p.state
end

"""
    struct RandomProjection
        w::Array{Float64, 2}
"""
struct RandomProjection
    w::Array{Float64, 2}
end
export RandomProjection
preprocessstate(p::RandomProjection, s) = p.w * s

"""
    struct SparseRandomProjection
        w::Array{Float64, 2}
        b::Array{Float64, 1}
"""
struct SparseRandomProjection
    w::Array{Float64, 2}
    b::Array{Float64, 1}
end
export SparseRandomProjection
preprocessstate(p::SparseRandomProjection, s) = clamp.(p.w * s + p.b, 0, Inf)

"""
    struct ImagePreprocessor
        size
        chain

Use `chain` to preprocess a grayscale or color image of `size = (width, height)`.

Example:
```
p = ImagePreprocessor((100, 100), 
                      [ImageResizeNearestNeighbour((50, 80)),
                       ImageCrop(1:30, 10:80),
                       x -> x ./ 256])
x = rand(UInt8, 100, 100)
s = ReinforcementLearning.preprocessstate(p, x)
```
"""
@with_kw struct ImagePreprocessor{Ts}
    size::Ts
    chain::Array{Any, 1}
end
preprocessstate(p::ImagePreprocessor, s) = foldl((x, p) -> p(x),
                                                 p.chain, 
                                                 init = reshape(s, p.size))

"""
    struct ImageResizeNearestNeighbour
        outdim::Tuple{Int64, Int64}

Resize any image to `outdim = (width, height)` by nearest-neighbour
interpolation (i.e. subsampling).

Example:
```
r = ImageResizeNearestNeighbour((50, 50))
r(rand(200, 200))
r(rand(UInt8, 3, 100, 100))
```
"""
struct ImageResizeNearestNeighbour
    outdim::Tuple{Int64, Int64}
end
function (r::ImageResizeNearestNeighbour)(x)
    indim = size(x)
    xidx = round.(Int64, collect(1:r.outdim[1]) .* indim[end - 1]/r.outdim[1])
    yidx = round.(Int64, collect(1:r.outdim[2]) .* indim[end]/r.outdim[2])
    length(indim) > 2 ? x[:, xidx, yidx] : x[xidx, yidx]
end

"""
    struct ImageResizeBilinear
        outdim::Tuple{Int64, Int64}

Resize any image to `outdim = (width, height)` with bilinear interpolation.

Example:
```
r = ImageResizeBilinear((50, 50))
r(rand(200, 200))
r(rand(UInt8, 3, 100, 100))
```
"""
struct ImageResizeBilinear
    outdim::Tuple{Int64, Int64}
end
for N in 2:3
    @eval @__MODULE__() function (p::ImageResizeBilinear)(x::Array{T, $N}) where T
        indim = size(x)
        sx, sy = (indim[end-1] - 1)/(p.outdim[1] + 1), (indim[end] - 1)/(p.outdim[2] + 1)
        $(N == 2 ? :(y = zeros(p.outdim[1], p.outdim[2])) : 
                   :(y = zeros(3, p.outdim[1], p.outdim[2])))
        for i in 1:p.outdim[1]
            for j in 1:p.outdim[2]
                r = floor(Int64, i*sx)
                c = floor(Int64, j*sy)
                dr = i*sx - r; dc = j*sy - c
                $(N == 2 ? :(
                y[i, j] = x[r + 1, c + 1] * (1 - dr) * (1 - dc) + 
                          x[r + 2, c + 1] * dr * (1 - dc) + 
                          x[r + 1, c + 2] * (1 - dr) * dc + 
                          x[r + 2, c + 2] * dr * dc) : 
                           :(
                y[:, i, j] .= x[:, r + 1, c + 1] * (1 - dr) * (1 - dc) .+ 
                              x[:, r + 2, c + 1] * dr * (1 - dc) .+ 
                              x[:, r + 1, c + 2] * (1 - dr) * dc .+ 
                              x[:, r + 2, c + 2] * dr * dc))
            end
        end
        y
    end
end

"""
    struct ImageCrop
        xidx::UnitRange{Int64}
        yidx::UnitRange{Int64}

Select indices `xidx` and `yidx` from a 2 or 3 dimensional array.

Example:
```
c = ImageCrop(2:5, 3:2:9)
c([10i + j for i in 1:10, j in 1:10])
```
"""
struct ImageCrop{Tx, Ty}
    xidx::Tx
    yidx::Ty
end
(c::ImageCrop)(x::Array{T, 2}) where T = x[c.xidx, c.yidx]
(c::ImageCrop)(x::Array{T, 3}) where T = x[:, c.xidx, c.yidx]

export ImagePreprocessor, ImageCrop, ImageResizeNearestNeighbour,
ImageResizeBilinear, togpu
