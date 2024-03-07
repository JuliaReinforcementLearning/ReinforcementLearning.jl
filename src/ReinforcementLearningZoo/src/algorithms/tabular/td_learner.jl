export TDLearner

using LinearAlgebra: dot
using Distributions: pdf
import Base.push!

"""
    TDLearner(;approximator, γ=1.0, method, n=0)

Use temporal-difference method to estimate state value or state-action value.

# Fields
- `approximator` can be either a
  `TabularQApproximator`, `LinearQApproximator`, `TabularVApproximator` or `LinearVApproximator`.
- `γ=1.0`, discount rate.
- `method`: only `:SARS` (Q-learning) is supported for the time being.
- `n=0`: the number of time steps used minus 1.
"""
Base.@kwdef struct TDLearner{M} <: AbstractLearner
    approximator::Approximator
    γ::Float64 = 1.0
    n::Int = 0

    function TDLearner(approximator::Approximator, method::Symbol; γ=1.0, n=0)
        if method ∉ [:SARS]
            @error "Method $method is not supported"
        else
            new{method}(approximator, γ, n)
        end
    end
end

RLCore.forward(L::TDLearner, s) = RLCore.forward(L.approximator, s)
RLCore.forward(L::TDLearner, s, a) = RLCore.forward(L.approximator, s, a)

function _optimise!(
    n::I1,
    γ::F,
    app::Approximator{Ar},
    s::I2,
    s_next::I2,
    a::I3,
    r::F,
) where {I1<:Number,I2<:Number,I3<:Number,Ar<:AbstractArray,F<:AbstractFloat}
    α = app.optimizer.eta
    Q!(app, s, s_next, a, α, r, γ)
end

function RLBase.optimise!(
    L::TDLearner,
    t::@NamedTuple{state::I1, next_state::I1, action::I2, reward::F2, terminal::Bool},
) where {I1<:Number,I2<:Number,F2<:AbstractFloat}
    _optimise!(L.n, L.γ, L.approximator, t.state, t.next_state, t.action, t.reward)
end

function RLBase.priority(L::TDLearner{:SARS}, transition)
    s, a, r, d, s′ = transition
    γ, Q = L.γ, L.approximator
    if d
        Δ = (r - RLCore.forward(Q, s, a))
    else
        Δ = (r + γ * RLCore.forward(Q, s′) - RLCore.forward(Q, s, a))
    end
    Δ = [Δ]  # must be broadcastable in Flux.Optimise
    Flux.Optimise.apply!(Q.optimizer, (s, a), Δ)
    abs(Δ[])
end


