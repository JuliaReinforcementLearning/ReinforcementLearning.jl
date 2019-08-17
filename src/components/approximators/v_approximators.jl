export LinearVApproximator, TabularVApproximator

#####
# LinearVApproximator
#####

using LinearAlgebra:dot

"""
Using a matrix `features` to represent each state along with a vector of `weights`.
"""
struct LinearVApproximator <: AbstractVApproximator{Int}
    features::Array{Float64, 2}
    weights::Vector{Float64}
end

# TODO: support Vector
(LinearVApproximator::LinearVApproximator)(s::Int) = @views dot(LinearVApproximator.features[s, :], LinearVApproximator.weights)

function update!(LinearVApproximator::LinearVApproximator, correction::Pair)
    s, e = correction
    for i in 1:length(LinearVApproximator.weights)
        LinearVApproximator.weights[i] += LinearVApproximator.features[s, i] * e
    end
end

#####
# TabularVApproximator
#####

"""
Using a `table` of type `Vector{Float64}` to record the state values.
"""
struct TabularVApproximator <: AbstractVApproximator{Int}
    table::Vector{Float64}
end

TabularVApproximator(ns::Int, init::Float64=0.) = TabularVApproximator(fill(init, ns))

(v::TabularVApproximator)(s::Int) = v.table[s]

function update!(v::TabularVApproximator, correction::Pair{Int, Float64})
    s, e = correction
    v.table[s] += e
end
