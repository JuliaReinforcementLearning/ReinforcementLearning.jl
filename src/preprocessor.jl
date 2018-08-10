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
