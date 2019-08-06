export TDLearner, update!

using .Utils: discount_rewards, discount_rewards_reduced

"""
    TDLearner(approximator::Tapp, γ::Float64, α::Float64; n::Int=0) where {Tapp<:VApproximator}
    TDLearner(approximator::Tapp, γ::Float64, α::Float64; n::Int=0, method::Symbol=:SARSA) where {Tapp<:QApproximator} 

The `TDLearner`(Temporal Difference Learner) use the latest `n` step experiences to update the `approximator`.
Note that `n` starts with `0`, which means looking forward for the next `n` steps.
`γ` is the discount rate of experience.
`α` is the learning rate.

For [`VApproximator`](@ref), the only supported update method is `:SRS`, which means
only **S**tates, **R**ewards and next_**S**ates are used to update the `approximator`.

For [`QApproximator`](@ref), the following methods are supported:

- `:SARSA`
- `:ExpectedSARSA`
"""
struct TDLearner{Tapp <: AbstractApproximator, method} <: AbstractLearner{Tapp}
    approximator::Tapp
    γ::Float64
    α::Float64
    n::Int

    TDLearner(approximator::Tapp, γ::Float64, α::Float64; n::Int=0) where {Tapp<:VApproximator} = new{Tapp, :SRS}(approximator, π, γ, α, n)

    function TDLearner(approximator::Tapp, γ::Float64, α::Float64; n::Int=0, method::Symbol=:SARSA) where {Tapp<:QApproximator} 
        supported_methods = [:SARSA, :ExpectedSARSA]
        !in(method, supported_methods) && throw(ArgumentError("Supported methods are $supported_methods , your input is $method"))
        new{Tapp, method}(approximator, γ, α, n)
    end
end

(learner::TDLearner)(s) = learner.approximator(s)
(learner::TDLearner{<:QApproximator})(s, a) = learner.approximator(s, a)

function update!(learner::TDLearner{<:VApproximator, :SRS}, states, rewards, terminals, next_states)
    n, γ, V, α = learner.n, learner.γ, learner.approximator, learner.α

    if terminals[end]
        @views gains = discount_rewards(rewards[max(end-n, 1):end], γ)  # n starts with 0
        for (i, G) in enumerate(gains)
            @views s = states[end-length(gains)+i]
            update!(V, s => α * (G - V(s)))
        end
    else
        if length(states) ≥ (n + 1)  # n starts with 0
            @views G = discount_rewards_reduced(rewards[end-n:end], γ) + γ^n * V(next_states[end])
            @views s = states[end-n]
            update!(V, s => α * (G - V(s)))
        end
    end
end

function update!(learner::TDLearner{<:QApproximator, :SARSA}, states, actions, rewards, terminals, next_states, next_actions)
    n, γ, Q, α = learner.n, learner.γ, learner.approximator, learner.α

    if terminals[end]
        @views gains = discount_rewards(rewards[max(end-n, 1):end], γ)  # n starts with 0
        for (i, G) in enumerate(gains)
            @views s, a = states[end-length(gains)+i], actions[end-length(gains)+i]
            update!(Q, (s, a) => α * (G - Q(s, a)))
        end
    else
        if length(states) ≥ (n + 1)  # n starts with 0
            @views s, a, s′, a′ = states[end-n], actions[end-n], next_states[end], next_actions[end]
            @views G = discount_rewards_reduced(rewards[end-n:end], γ) + γ^n * Q(s′, a′)
            update!(Q, (s, a) => α * (G - Q(s, a)))
        end
    end
end