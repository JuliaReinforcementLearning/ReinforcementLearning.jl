export LinearVApproximator, TabularVApproximator

#####
# LinearVApproximator
#####

using LinearAlgebra: dot

"""
Using a matrix `features` to represent each state along with a vector of `weights`.
"""
struct LinearVApproximator{N} <: AbstractVApproximator
    weights::Array{Float64,N}
end

# TODO: support Vector
(V::LinearVApproximator)(s) = dot(s, V.weights)

function update!(V::LinearVApproximator, correction::Pair)
    s, e = correction
    V.weights .+= s .* e
end

#####
# TabularVApproximator
#####

"""
Using a `table` of type `Vector{Float64}` to record the state values.
"""
struct TabularVApproximator <: AbstractVApproximator
    table::Vector{Float64}
end

TabularVApproximator(ns::Int, init::Float64 = 0.0) = TabularVApproximator(fill(init, ns))

(v::TabularVApproximator)(s::Int) = v.table[s]

function update!(v::TabularVApproximator, correction::Pair{Int,Float64})
    s, e = correction
    v.table[s] += e
end
