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

Q(app::TabularApproximator, s, a) = RLCore.forward(app, s, a)
Q(app::TabularApproximator, s) = RLCore.forward(app, s)

"""
    Q!(app::TabularApproximator, s::Int, s_plus_one::Int, a::Int, α::Float64, π_::Float64, γ::Float64)

Update the Q-value of the given state-action pair.
"""
function Q!(
    app::TabularApproximator,
    s::I1,
    s_plus_one::I2,
    a::I3,
    α::F1,
    π_::F2,
    γ::Float64,
) where {I1<:Integer,I2<:Integer,I3<:Integer,F1<:AbstractFloat,F2<:AbstractFloat}
    # Q-learning formula according to https://github.com/JuliaPOMDP/TabularTDLearning.jl/blob/25c4d3888e178c51ed1ff448f36b0fcaf7c1d8e8/src/q_learn.jl#LL63C26-L63C95
    q_value_updated = α * (π_ + γ * maximum(Q(app, s_plus_one)) - Q(app, s, a))
    app.model[a, s] += q_value_updated
    return Q(app, s, a)
end

function _optimise!(
    n::I1,
    γ::F,
    app::Approximator{Ar},
    s::I2,
    s_next::I2,
    a::I3,
    r::F,
) where {I1<:Number,I2<:Number,I3<:Number,Ar<:AbstractArray,F<:AbstractFloat}
    α = app.optimiser_state.eta
    Q!(app, s, s_next, a, α, r, γ)
end

function RLBase.optimise!(
    L::TDLearner,
    t::@NamedTuple{state::I1, next_state::I1, action::I2, reward::F2, terminal::Bool},
) where {I1<:Number,I2<:Number,F2<:AbstractFloat}
    _optimise!(L.n, L.γ, L.approximator, t.state, t.next_state, t.action, t.reward)
end

