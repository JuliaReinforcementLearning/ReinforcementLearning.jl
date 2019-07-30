export TilingsQ, update!

using .Utils:Tiling

"""
The only difference compared to [`TilingsV`](@ref) is that now the weight of each tiling is a matrix.
"""
struct TilingsQ{Tt<:Tiling} <: AbstractQApproximator{Vector{Float64}}
    tilings::Vector{Tt}
    weights::Vector{Array{Float64, 2}}
    TilingsQ(tilings::Vector{Tt}, nactions) where Tt<:Tiling = new{Tt}(tilings, [zeros(Float64, length(t), nactions) for t in tilings])
end

function (ts::TilingsQ)(s, a)
    v = 0.
    for (w, t) in zip(ts.weights, ts.tilings)
        v += w[encode(t, s), a]
    end
    v
end

function (ts::TilingsQ)(s) 
    dist = zeros(Float64, size(ts.weights[1],2))
    for (w,t) in zip(ts.weights, ts.tilings)
        dist .+= @view w[encode(t,s), :]
    end
    dist
end

function update!(ts::TilingsQ, correction::Pair)
    (s, a), e = correction
    for (w, t) in zip(ts.weights, ts.tilings)
        w[encode(t, s), a] += e
    end
end