export TDLearner, update!

using .Utils: discount_rewards, discount_rewards_reduced
using Flux.Optimise:Descent

"""
    TDLearner(approximator::Tapp, γ::Float64, optimizer::Float64; n::Int=0) where {Tapp<:AbstractVApproximator}
    TDLearner(approximator::Tapp, γ::Float64, optimizer::Float64; n::Int=0, method::Symbol=:SARSA) where {Tapp<:AbstractQApproximator} 

The `TDLearner`(Temporal Difference Learner) use the latest `n` step experiences to update the `approximator`.
Note that `n` starts with `0`, which means looking forward for the next `n` steps.
`γ` is the discount rate of experience.
`optimizer` is the learning rate.

For [`AbstractVApproximator`](@ref), the only supported update method is `:SRS`, which means
only **S**tates, **R**ewards and next_**S**ates are used to update the `approximator`.

For [`AbstractQApproximator`](@ref), the following methods are supported:

- `:SARSA`
- `:ExpectedSARSA`
"""
mutable struct TDLearner{Tapp <: AbstractApproximator, method, O} <: AbstractLearner
    approximator::Tapp
    γ::Float64
    optimizer::O
    n::Int

    function TDLearner(;approximator::Tapp, γ=1.0, optimizer=Descent(1.0), n=0, method::Symbol=:SARSA) where {Tapp<:AbstractQApproximator}
        if approximator isa AbstractVApproximator
            if method != :SRS
                throw(ArgumentError("method [$method] is unsupported for a value approximator"))
            end
        elseif approximator isa AbstractQApproximator
            if !in(method, [:SARSA, :ExpectedSARSA])
                throw(ArgumentError("Supported methods are $supported_methods , your input is $method"))
            end
        else
            throw(ArgumentError("unknown approximator"))
        end
        new{Tapp, method, typeof(optimizer)}(approximator, γ, optimizer, n)
    end
end

function update!(learner::TDLearner{<:AbstractVApproximator, :SRS}, states, rewards, terminals, next_states)
    n, γ, V, optimizer = learner.n, learner.γ, learner.approximator, learner.optimizer

    if length(terminals) > 0 && terminals[end]
        @views gains = discount_rewards(rewards[max(end-n, 1):end], γ)  # n starts with 0
        for (i, G) in enumerate(gains)
            @views s = states[end-length(gains)+i]
            update!(V, s => apply!(optimizer, s, G - V(s)))
        end
    else
        if length(states) ≥ (n + 1)  # n starts with 0
            @views G = discount_rewards_reduced(rewards[end-n:end], γ) + γ^n * V(next_states[end])
            @views s = states[end-n]
            update!(V, s => apply!(optimizer, s, G - V(s)))
        end
    end
end

function update!(learner::TDLearner{<:AbstractQApproximator, :SARSA}, states, actions, rewards, terminals, next_states, next_actions)
    n, γ, Q, optimizer = learner.n, learner.γ, learner.approximator, learner.optimizer

    if length(terminals) > 0 && terminals[end]
        @views gains = discount_rewards(rewards[max(end-n, 1):end], γ)  # n starts with 0
        for (i, G) in enumerate(gains)
            @views s, a = states[end-length(gains)+i], actions[end-length(gains)+i]
            update!(Q, (s, a) => apply!(optimizer, (s, a), G - Q(s, a)))
        end
    else
        if length(states) ≥ (n + 1)  # n starts with 0
            @views s, a, s′, a′ = states[end-n], actions[end-n], next_states[end], next_actions[end]
            @views G = discount_rewards_reduced(rewards[end-n:end], γ) + γ^n * Q(s′, a′)
            update!(Q, (s, a) => apply!(optimizer, (s, a), G - Q(s, a)))
        end
    end
end