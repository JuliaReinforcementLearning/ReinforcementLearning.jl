export LinearVApproximator, LinearQApproximator

using LinearAlgebra: dot

"""
    LinearQApproximator(weights::Vector{Float64}, feature_func::F, actions::Vector{Int}) -> LinearQApproximator{F}

Use weighted sum to represent the estimation given a state and an action.

# Fields

- `weights::Vector{Float64}`: the weight of each feature.
- `feature_func::Function`: decide how to generate a feature vector of `length(weights)` given a state and an action as parameters.
- `actions::Vector{Int}`: all possible actions.

See also [`LinearVApproximator`](@ref).
"""
Base.@kwdef struct LinearQApproximator{F} <: AbstractQApproximator
    weights::Vector{Float64}
    feature_func::F
    actions::Vector{Int}
end

(Q::LinearQApproximator)(s, a::Int) = dot(Q.weights, Q.feature_func(s, a))

(Q::LinearQApproximator)(s) = [Q(s, a) for a in Q.actions]

function update!(Q::LinearQApproximator, correction::Pair)
    (s, a), e = correction
    xs = Q.feature_func(s, a)
    Q.weights .+= xs .* e
end