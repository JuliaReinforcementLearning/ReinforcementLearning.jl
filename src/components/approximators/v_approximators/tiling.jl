export TilingsV, update!

using .Utils:Tiling, encode

"""
    TilingsV{Tt<:Tiling} <: AbstractVApproximator{Vector{Float64}}
    TilingsV(tilings::Vector{Tt}) where Tt<:Tiling

Using a vector of `tilings` to encode state. Each tiling has an independent weight.

See more details at Section (9.5.4) on Page 217 of the book *Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.*
"""
struct TilingsV{Tt<:Tiling} <: AbstractVApproximator{Vector{Float64}}
    tilings::Vector{Tt}
    weights::Vector{Vector{Float64}}
    TilingsV(tilings::Vector{Tt}) where Tt<:Tiling = new{Tt}(tilings, [zeros(Float64, length(t)) for t in tilings])
end

function (ts::TilingsV)(s)
    v = 0.
    for (w, t) in zip(ts.weights, ts.tilings)
        v += w[encode(t, s)]
    end
    v
end

function update!(ts::TilingsV, correction::Pair)
    s, e = correction
    for i in 1:length(ts.tilings)
        ts.weights[i][encode(ts.tilings[i], s)] += e
    end
end