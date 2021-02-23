export LinearApproximator, LinearVApproximator, LinearQApproximator

using LinearAlgebra: dot

struct LinearApproximator{N,O} <: AbstractApproximator
    weights::Array{Float64,N}
    optimizer::O
end

#####
# LinearVApproximator
#####

const LinearVApproximator = LinearApproximator{1}

LinearVApproximator(; n, init = 0.0, opt = Descent(1.0)) =
    LinearApproximator(fill(init, n), opt)

(V::LinearVApproximator)(s) = dot(s, V.weights)

function RLBase.update!(V::LinearVApproximator, correction::Pair)
    w = V.weights
    s, Δ = correction
    w̄ = s .* Δ
    Flux.Optimise.update!(V.optimizer, w, w̄)
end

#####
# LinearQApproximator
#####

const LinearQApproximator = LinearApproximator{2}

LinearQApproximator(; n_state, n_action, init = 0.0, opt = Descent(1.0)) =
    LinearApproximator(fill(init, n_state, n_action), opt)

(Q::LinearQApproximator)(s) = [dot(s, c) for c in eachcol(Q.weights)]
(Q::LinearQApproximator)(s, a) = dot(s, @view(Q.weights[:, a]))

function RLBase.update!(Q::LinearQApproximator, correction::Pair)
    (s, a), Δ = correction
    @views w = Q.weights[:, a]
    w̄ = s .* Δ
    Flux.Optimise.update!(Q.optimizer, w, w̄)
end
