export LinearVApproximator, TabularVApproximator

#####
# LinearVApproximator
#####

using LinearAlgebra: dot

"""
    LinearVApproximator(weights::Array{Float64, N}) -> LinearVApproximator{N}

Use the weighted sum to represent the estimation of a state.
The state is expected to have the same length with `weights`.

See also [`LinearQApproximator`](@ref)
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
    TabularVApproximator(table) -> TabularVApproximator
    TabularVApproximator(ns::Int, init::Float64=0.0) -> TabularVApproximator

Use a `table` of type `Vector{Float64}` of length `ns` to record the state values.
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
