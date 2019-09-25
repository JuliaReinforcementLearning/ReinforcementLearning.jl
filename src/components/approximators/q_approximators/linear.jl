export LinearVApproximator, LinearQApproximator

using LinearAlgebra: dot

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