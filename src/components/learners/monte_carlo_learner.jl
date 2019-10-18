export MonteCarloLearner,
       FIRST_VISIT,
       EVERY_VISIT,
       NO_SAMPLING,
       ORDINARY_IMPORTANCE_SAMPLING,
       WEIGHTED_IMPORTANCE_SAMPLING

using .Utils: CachedSampleAvg
using StatsBase: countmap

abstract type AbstractVisitType end
struct FirstVisit <: AbstractVisitType end
struct EveryVisit <: AbstractVisitType end
const FIRST_VISIT = FirstVisit()
const EVERY_VISIT = EveryVisit()

abstract type AbstractSamplingType end
struct NoSampling <: AbstractSamplingType end
struct OrdinaryImportanceSampling <: AbstractSamplingType end
struct WeightedImportanceSampling <: AbstractSamplingType end
const NO_SAMPLING = NoSampling()
const ORDINARY_IMPORTANCE_SAMPLING = OrdinaryImportanceSampling()
const WEIGHTED_IMPORTANCE_SAMPLING = WeightedImportanceSampling()

"""
    MonteCarloLearner(; approximator::A, γ = 1.0, α = 1.0, kind = FIRST_VISIT, sampling = NO_SAMPLING, returns = CachedSampleAvg())
"""
struct MonteCarloLearner{T,A,R,S} <: AbstractLearner
    approximator::A
    γ::Float64
    α::Float64
    returns::R

    MonteCarloLearner(
        ;
        approximator::A,
        γ = 1.0,
        α = 1.0,
        kind = FIRST_VISIT,
        sampling = NO_SAMPLING,
        returns = CachedSampleAvg(),
    ) where {A} =
        new{typeof(kind),A,typeof(returns),typeof(sampling)}(approximator, γ, α, returns)
end

(learner::MonteCarloLearner)(s) = learner.approximator(s)
(learner::MonteCarloLearner)(obs::Observation) = learner.approximator(get_state(obs))
(learner::MonteCarloLearner)(obs::Observation, a) = learner.approximator(get_state(s), a)

function update!(
    learner::MonteCarloLearner{<:FirstVisit,<:AbstractVApproximator,<:Any,<:NoSampling},
    transitions,
)
    states, rewards = transitions.states, transitions.rewards
    V, γ, α, Returns, G, T = learner.approximator,
        learner.γ,
        learner.α,
        learner.returns,
        0.0,
        length(states)
    seen_states = countmap(states)

    for t = T:-1:1
        S, R = states[t], rewards[t]
        G = γ * G + R
        if seen_states[S] == 1  # first visit
            update!(V, S => α * (Returns(S, G) - V(S)))
            delete!(seen_states, S)
        else
            seen_states[S] -= 1
        end
    end
end

function update!(
    learner::MonteCarloLearner{
        <:FirstVisit,
        <:AbstractVApproximator,
        <:CachedSampleAvg,
        <:OrdinaryImportanceSampling,
    },
    transitions,
    weights,
)
    states, rewards = transitions.states, transitions.rewards
    V, γ, α, Returns, G, ρ, T = learner.approximator,
        learner.γ,
        learner.α,
        learner.returns,
        0.0,
        1.0,
        length(states)
    seen_states = countmap(states)

    for t = T:-1:1
        S, R = states[t], rewards[t]
        G = γ * G + R
        ρ *= weights[t]
        if seen_states[S] == 1  # first visit
            update!(V, S => α * (Returns(S, ρ * G) - V(S)))
            delete!(seen_states, S)
        else
            seen_states[S] -= 1
        end
    end
end

function update!(
    learner::MonteCarloLearner{
        <:FirstVisit,
        <:AbstractVApproximator,
        <:Tuple{CachedSum,CachedSum},
        <:WeightedImportanceSampling,
    },
    transitions,
    weights,
)
    states, rewards = transitions.states, transitions.rewards
    V, γ, α, (G_cached, ρ_cached), G, ρ, T = learner.approximator,
        learner.γ,
        learner.α,
        learner.returns,
        0.0,
        1.0,
        length(states)
    seen_states = countmap(states)

    for t = T:-1:1
        S, R = states[t], rewards[t]
        G = γ * G + R
        ρ *= weights[t]
        if seen_states[S] == 1  # first visit
            numerator = G_cached(S, ρ * G)
            denominator = ρ_cached(S, ρ)
            val = denominator == 0 ? 0 : numerator / denominator
            update!(V, S => α * (val - V(S)))
            delete!(seen_states, S)
        else
            seen_states[S] -= 1
        end
    end
end

function update!(
    learner::MonteCarloLearner{<:EveryVisit,<:AbstractVApproximator},
    transitions,
)
    states, rewards = transitions.states, transitions.rewards
    α, γ, V, Returns, G = learner.α, learner.γ, learner.approximator, learner.returns, 0.0
    for (s, r) in Iterators.reverse(zip(states, rewards))
        G = γ * G + r
        update!(V, s => α * (Returns(s, G) - V(s)))
    end
end

function update!(
    learner::MonteCarloLearner{<:FirstVisit,<:AbstractQApproximator},
    transitions,
)
    states, actions, rewards = transitions.states, transitions.actions, transitions.rewards
    α, γ, Q, Returns, G, T = learner.α,
        learner.γ,
        learner.approximator,
        learner.returns,
        0.0,
        length(states)
    seen_pairs = countmap(zip(states, actions))

    for t = T:-1:1
        S, A, R = states[t], actions[t], rewards[t]
        pair = (S, A)
        G = γ * G + R
        if seen_pairs[pair] == 1  # first visit
            update!(Q, pair => α * (Returns(pair, G) - Q(S, A)))
            delete!(seen_pairs, pair)
        else
            seen_pairs[pair] -= 1
        end
    end
end

function update!(
    learner::MonteCarloLearner{<:EveryVisit,<:AbstractQApproximator},
    transitions,
)
    states, actions, rewards = transitions.states, transitions.actions, transitions.rewards
    α, γ, Q, Returns, G = learner.α, learner.γ, learner.approximator, learner.returns, 0.0
    for (s, a, r) in Iterators.reverse(zip(states, actions, rewards))
        G = γ * G + r
        update!(Q, (s, a) => α * (Returns((s, a), G) - Q(s, a)))
    end
end

function extract_transitions(
    buffer::EpisodeTurnBuffer,
    ::MonteCarloLearner{T,A},
) where {T,A<:AbstractQApproximator}
    if isfull(buffer)
        @views (
            states = state(buffer)[1:end-1],
            actions = action(buffer)[1:end-1],
            rewards = reward(buffer)[2:end],
        )
    else
        nothing
    end
end
