export LinearVApproximator, LinearQApproximator

using LinearAlgebra: dot

struct LinearQApproximator{N} <: AbstractQApproximator
    weights::Array{Float64, N}
end

(Q::LinearQApproximator{N})(s, a::Int) where N = dot(s, selectdim(Q.weights, N, a))

(Q::LinearQApproximator{N})(s) where N = [dot(s, selectdim(Q.weights, N, a)) for a in axes(Q.weights, N)]

function update!(Q::LinearQApproximator{N}, correction::Pair) where N
    (s, a), e = correction
    selectdim(Q.weights, N, a) .+= s .* e
end