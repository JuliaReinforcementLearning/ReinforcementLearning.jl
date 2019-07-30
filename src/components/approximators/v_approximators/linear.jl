export LinearV, update!

using LinearAlgebra:dot

"""
    struct LinearV <: AbstractVApproximator{Int}
        features::Array{Float64, 2}
        weights::Vector{Float64}
    end

Using a matrix `features` to represent each state along with a vector of `weights`.

See more details at Section (9.4) on Page 205 of the book *Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.*
"""
struct LinearV <: AbstractVApproximator{Int}
    features::Array{Float64, 2}
    weights::Vector{Float64}
end

(linearV::LinearV)(s::Int) = @views dot(linearV.features[s, :], linearV.weights)

function update!(linearV::LinearV, correction::Pair)
    s, e = correction
    for i in 1:length(linearV.weights)
        linearV.weights[i] += linearV.features[s, i] * e
    end
end