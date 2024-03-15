export TDLearner

using LinearAlgebra: dot
using Distributions: pdf
import Base.push!

using ReinforcementLearningCore: AbstractLearner, TabularApproximator
using Flux

"""
    TDLearner(;approximator, γ=1.0, method, n=0)

Use temporal-difference method to estimate state value or state-action value.

# Fields
- `approximator` is `<:TabularApproximator`.
- `γ=1.0`, discount rate.
- `method`: only `:SARS` (Q-learning) is supported for the time being.
- `n=0`: the number of time steps used minus 1.
"""
Base.@kwdef mutable struct TDLearner{M,A} <: AbstractLearner where {A<:TabularApproximator,M<:Symbol}
    approximator::A
    γ::Float64 = 1.0
    n::Int = 0

    function TDLearner(approximator::A, method::Symbol; γ=1.0, n=0) where {A<:TabularApproximator}
        if method ∉ [:SARS]
            @error "Method $method is not supported"
        else
            new{method, A}(approximator, γ, n)
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
    approx::TabularApproximator,
    s::I1,
    s_plus_one::I2,
    a::I3,
    π_::F1,
    γ::Float64, # discount factor
) where {I1<:Integer,I2<:Integer,I3<:Integer,F1<:AbstractFloat}
    # Q-learning formula following https://github.com/JuliaPOMDP/TabularTDLearning.jl/blob/25c4d3888e178c51ed1ff448f36b0fcaf7c1d8e8/src/q_learn.jl#LL63C26-L63C95
    # Terminology following https://en.wikipedia.org/wiki/Q-learning
    estimate_optimal_future_value = maximum(Q(approx, s_plus_one))
    current_value = Q(approx, s, a)
    raw_q_value = (π_ + γ * estimate_optimal_future_value - current_value) # Discount factor γ is applied here
    q_value_updated = Flux.Optimise.apply!(approx.optimiser_state, :learning, [raw_q_value])[] # adust according to optimiser learning rate
    approx.model[a, s] += q_value_updated
    return Q(approx, s, a)
end

function _optimise!(
    n::I1,
    γ::F,
    approx::Approximator{Ar},
    s::I2,
    s_next::I2,
    a::I3,
    r::F,
) where {I1<:Number,I2<:Number,I3<:Number,Ar<:AbstractArray,F<:AbstractFloat}
    Q!(approx, s, s_next, a, r, γ)
end

function RLBase.optimise!(
    L::TDLearner,
    t::@NamedTuple{state::I1, next_state::I1, action::I2, reward::F2, terminal::Bool},
) where {I1<:Number,I2<:Number,F2<:AbstractFloat}
    _optimise!(L.n, L.γ, L.approximator, t.state, t.next_state, t.action, t.reward)
end
