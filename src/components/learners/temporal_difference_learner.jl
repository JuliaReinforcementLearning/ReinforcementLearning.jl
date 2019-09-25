export TDLearner, DoubleLearner, DifferentialTDLearner, TDλReturnLearner

using .Utils: discount_rewards, discount_rewards_reduced
using Flux.Optimise: Descent
using LinearAlgebra: dot

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

- `:SARS` (aka Q-Learning)
- `:SARSA`
- `:ExpectedSARSA`
"""
mutable struct TDLearner{Tapp<:AbstractApproximator,method,O} <: AbstractLearner
    approximator::Tapp
    γ::Float64
    optimizer::O
    n::Int

    function TDLearner(
        ;
        approximator::Tapp,
        γ = 1.0,
        optimizer = Descent(1.0),
        n = 0,
        method::Symbol = :SARSA,
    ) where {Tapp<:AbstractApproximator}
        if approximator isa AbstractVApproximator
            if method != :SRS
                throw(ArgumentError("method [$method] is unsupported for a value approximator"))
            end
        elseif approximator isa AbstractQApproximator
            if !in(method, [:SARSA, :SARS, :ExpectedSARSA])
                throw(ArgumentError("Supported methods are $supported_methods , your input is $method"))
            end
        else
            throw(ArgumentError("unknown approximator"))
        end
        new{Tapp,method,typeof(optimizer)}(approximator, γ, optimizer, n)
    end
end

Base.@kwdef struct DoubleLearner{T<:TDLearner} <: AbstractLearner
    L1::T
    L2::T
end

#####
# DifferentialTDLearner
#####

Base.@kwdef mutable struct DifferentialTDLearner{A<:AbstractApproximator} <: AbstractLearner
    approximator::A
    α::Float64
    β::Float64
    R̄::Float64 = 0.0
    n::Int = 0
end

function update!(learner::DifferentialTDLearner, transition)
    states, actions, rewards, terminals, next_states, next_actions = transition
    n, α, β, Q = learner.n, learner.α, learner.β, learner.approximator
    if length(states) ≥ n + 1
        s, a = states[1], actions[1]
        s′, a′ = next_states[end], next_actions[end]
        δ = sum(r -> r - learner.R̄, rewards) + Q(s′, a′) - Q(s, a)
        learner.R̄ += β * δ
        update!(Q, (s, a) => α * δ)
    end
end

extract_transitions(buffer, learner::DoubleLearner) =
    extract_transitions(buffer, learner.L1)

function update!(learner::DoubleLearner, args...)
    rand(Bool) ? update!(learner.L1, learner.L2, args...) :
    update!(learner.L2, learner.L1, args...)
end

(learner::DoubleLearner)(obs::Observation) = learner.L1(obs) .+ learner.L2(obs)

update!(learner::TDLearner{<:AbstractVApproximator,:SRS}, transitions) =
    update!(learner, transitions, nothing)

function update!(learner::TDLearner{<:AbstractVApproximator,:SRS}, transitions, weights)
    states, rewards, terminals, next_states = transitions.states,
        transitions.rewards,
        transitions.terminals,
        transitions.next_states
    n, γ, V, optimizer = learner.n, learner.γ, learner.approximator, learner.optimizer

    if length(terminals) > 0 && terminals[end]
        @views gains = discount_rewards(rewards[max(end - n, 1):end], γ)  # n starts with 0
        cum_weights = isnothing(weights) ? nothing : cumprod!(reverse(weights))
        for (i, G) in enumerate(gains)
            @views s = states[end-length(gains)+i]
            if isnothing(cum_weights)
                update!(V, s => apply!(optimizer, s, G - V(s)))
            else
                update!(V, s => apply!(optimizer, s, cum_weights[i] * (G - V(s))))
            end
        end
    else
        if length(states) ≥ (n + 1)  # n starts with 0
            @views G = discount_rewards_reduced(rewards[end-n:end], γ) +
                       γ^(n + 1) * V(next_states[end])
            @views s = states[end-n]
            w = isnothing(weights) ? 1.0 : reduce(*, weights)
            update!(V, s => apply!(optimizer, s, w * (G - V(s))))
        end
    end
end

function extract_transitions(
    buffer::EpisodeTurnBuffer,
    learner::TDLearner{<:AbstractVApproximator,:SRS},
)
    n = learner.n
    if length(buffer) > 0
        @views (
            states = state(buffer)[max(1, end - n - 1):end-1],
            rewards = reward(buffer)[max(1, end - n - 1)+1:end],
            terminals = terminal(buffer)[max(1, end - n - 1)+1:end],
            next_states = state(buffer)[max(1, end - n - 1)+1:end],
        )
    else
        nothing
    end
end

function update!(learner::TDLearner{<:AbstractQApproximator,:SARSA}, transitions)
    states, actions, rewards, terminals, next_states, next_actions = transitions
    n, γ, Q, optimizer = learner.n, learner.γ, learner.approximator, learner.optimizer

    if length(terminals) > 0 && terminals[end]
        @views gains = discount_rewards(rewards[max(end - n, 1):end], γ)  # n starts with 0
        for (i, G) in enumerate(gains)
            @views s, a = states[end-length(gains)+i], actions[end-length(gains)+i]
            update!(Q, (s, a) => apply!(optimizer, (s, a), G - Q(s, a)))
        end
    else
        if length(states) ≥ (n + 1)  # n starts with 0
            @views s, a, s′, a′ = states[end-n],
                actions[end-n],
                next_states[end],
                next_actions[end]
            @views G = discount_rewards_reduced(rewards[end-n:end], γ) +
                       γ^(n + 1) * Q(s′, a′)
            update!(Q, (s, a) => apply!(optimizer, (s, a), G - Q(s, a)))
        end
    end
end

function extract_transitions(
    buffer::EpisodeTurnBuffer,
    learner::Union{TDLearner{<:AbstractQApproximator,:SARSA},DifferentialTDLearner},
)
    n = learner.n
    if length(buffer) > 0
        @views (
            states = state(buffer)[max(1, end - n - 1):end-1],
            actions = action(buffer)[max(1, end - n - 1):end-1],
            rewards = reward(buffer)[max(1, end - n - 1)+1:end],
            terminals = terminal(buffer)[max(1, end - n - 1)+1:end],
            next_states = state(buffer)[max(1, end - n - 1)+1:end],
            next_actions = action(buffer)[max(1, end - n - 1)+1:end],
        )
    else
        nothing
    end
end

function update!(learner::TDLearner{<:AbstractQApproximator,:ExpectedSARSA}, transitions)
    states, actions, rewards, terminals, next_states, probs_of_a′ = transitions
    n, γ, Q, optimizer = learner.n, learner.γ, learner.approximator, learner.optimizer

    if length(terminals) > 0 && terminals[end]
        @views gains = discount_rewards(rewards[max(end - n, 1):end], γ)  # n starts with 0
        for (i, G) in enumerate(gains)
            @views s, a = states[end-length(gains)+i], actions[end-length(gains)+i]
            update!(Q, (s, a) => apply!(optimizer, (s, a), G - Q(s, a)))
        end
    else
        if length(states) ≥ (n + 1)  # n starts with 0
            @views s, a, s′ = states[end-n], actions[end-n], next_states[end]
            @views G = discount_rewards_reduced(rewards[end-n:end], γ) +
                       γ^(n + 1) * dot(Q(s′), probs_of_a′)
            update!(Q, (s, a) => apply!(optimizer, (s, a), G - Q(s, a)))
        end
    end
end

function update!(learner::TDLearner{<:AbstractQApproximator,:SARS}, transitions)
    states, actions, rewards, terminals, next_states = transitions
    n, γ, Q, optimizer = learner.n, learner.γ, learner.approximator, learner.optimizer

    if length(terminals) > 0 && terminals[end]
        @views gains = discount_rewards(rewards[max(end - n, 1):end], γ)  # n starts with 0
        for (i, G) in enumerate(gains)
            @views s, a = states[end-length(gains)+i], actions[end-length(gains)+i]
            update!(Q, (s, a) => apply!(optimizer, (s, a), G - Q(s, a)))
        end
    else
        if length(states) ≥ (n + 1)  # n starts with 0
            @views s, a, s′ = states[end-n], actions[end-n], next_states[end]
            @views G = discount_rewards_reduced(rewards[end-n:end], γ) +
                       γ^(n + 1) * maximum(Q(s′))  # n starts with 0
            update!(Q, (s, a) => apply!(optimizer, (s, a), G - Q(s, a)))
        end
    end
end

function update!(
    learner::T,
    target_learner::T,
    transitions,
) where {T<:TDLearner{<:AbstractQApproximator,:SARS}}
    states, actions, rewards, terminals, next_states = transitions
    n, γ, Q, Qₜ, optimizer = learner.n,
        learner.γ,
        learner.approximator,
        target_learner.approximator,
        learner.optimizer

    if length(terminals) > 0 && terminals[end]
        @views gains = discount_rewards(rewards[max(end - n, 1):end], γ)  # n starts with 0
        for (i, G) in enumerate(gains)
            @views s, a = states[end-length(gains)+i], actions[end-length(gains)+i]
            update!(Q, (s, a) => apply!(optimizer, (s, a), G - Q(s, a)))
        end
    else
        if length(states) ≥ (n + 1)  # n starts with 0
            @views s, a, s′ = states[end-n], actions[end-n], next_states[end]
            @views G = discount_rewards_reduced(rewards[end-n:end], γ) +
                       γ^(n + 1) * Qₜ(s′, argmax(Q(s′)))
            update!(Q, (s, a) => apply!(optimizer, (s, a), G - Q(s, a)))
        end
    end
end

function priority(transition, learner::TDLearner{<:AbstractQApproximator,:SARS})
    s, a, r, d, s′ = transition
    γ, Q, opt = learner.γ, learner.approximator, learner.optimizer
    error = d ? apply!(opt, (s, a), r - Q(s, a)) :
            apply!(opt, (s, a), r + γ^(learner.n + 1) * maximum(Q(s′)) - Q(s, a))
    abs(error)
end

function update!(
    learner::TDLearner{<:AbstractQApproximator,:SARS},
    model::Union{TimeBasedSampleModel,ExperienceBasedSampleModel};
    plan_step = 1,
)
    for _ = 1:plan_step
        # @assert learner.n == 0 "n must be 0 here"
        transitions = extract_transitions(model, learner)
        if !isnothing(transitions)
            update!(learner, transitions)
        end
    end
end

function extract_transitions(
    model::Union{ExperienceBasedSampleModel,TimeBasedSampleModel},
    learner::TDLearner{<:AbstractQApproximator,:SARS},
)
    if length(model.experiences) > 0
        map(Base.vect, sample(model))
    else
        nothing
    end
end

function update!(
    learner::TDLearner{<:AbstractQApproximator,:SARS},
    model::PrioritizedSweepingSampleModel;
    plan_step = 1,
)
    for _ = 1:plan_step
        # @assert learner.n == 0 "n must be 0 here"
        transitions = extract_transitions(model, learner)
        if !isnothing(transitions)
            update!(learner, transitions)
            s, _, _, _, _ = transitions
            s = s[]  # length(s) is assumed to be 1
            for (s̄, ā, r̄, d̄) in model.predecessors[s]
                P = priority((s̄, ā, r̄, d̄, s), learner)
                if P ≥ model.θ
                    model.PQueue[(s̄, ā)] = P
                end
            end
        end
    end
end

function extract_transitions(
    model::PrioritizedSweepingSampleModel,
    learner::TDLearner{<:AbstractQApproximator,:SARS},
)
    if length(model.PQueue) > 0
        map(Base.vect, sample(model))
    else
        nothing
    end
end

function extract_transitions(
    buffer::EpisodeTurnBuffer,
    learner::TDLearner{<:AbstractQApproximator,:SARS},
)
    n = learner.n
    if length(buffer) > 0
        @views (
            states = state(buffer)[max(1, end - n - 1):end-1],
            actions = action(buffer)[max(1, end - n - 1):end-1],
            rewards = reward(buffer)[max(1, end - n - 1)+1:end],
            terminals = terminal(buffer)[max(1, end - n - 1)+1:end],
            next_states = state(buffer)[max(1, end - n - 1)+1:end],
        )
    else
        nothing
    end
end

#####
# TDλReturnLearner
#####
Base.@kwdef struct TDλReturnLearner{Tapp<:AbstractApproximator} <: AbstractLearner
    approximator::Tapp
    γ::Float64 = 1.0
    α::Float64
    λ::Float64
end

function extract_transitions(
    buffer::EpisodeTurnBuffer,
    learner::TDλReturnLearner{<:AbstractVApproximator},
)
    if isfull(buffer)
        @views (
            states = state(buffer)[1:end-1],
            rewards = reward(buffer)[2:end],
            terminals = terminal(buffer)[2:end],
            next_states = state(buffer)[2:end],
        )
    else
        nothing
    end
end

function update!(learner::TDλReturnLearner, transition)
    λ, γ, V, α = learner.λ, learner.γ, learner.approximator, learner.α
    states, rewards, terminals, next_states = transition
    T = length(states)
    for t = 1:T
        G = 0.0
        for n = 1:(T-t)
            G += λ^(n - 1) *
                 (discount_rewards_reduced(@view(rewards[t:t+n-1]), γ) +
                  γ^n * V(next_states[t+n-1]))
        end
        G *= 1 - λ
        G += λ^(T - t) *
             (discount_rewards_reduced(@view(rewards[t:T]), γ) +
              γ^(T - t + 1) * V(next_states[T]))
        sₜ = states[t]
        update!(V, sₜ => α * (G - V(sₜ)))
    end
end