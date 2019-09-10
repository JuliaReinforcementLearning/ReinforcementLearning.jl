export MonteCarloLearner, update!

using .Utils:CachedSampleAvg
using StatsBase:countmap

struct MonteCarloLearner{T, A, R} <: AbstractLearner
    approximator::A
    γ::Float64
    α::Float64
    returns::R

    MonteCarloLearner(app::A; γ=1., α=1., kind=:FirstVisit, returns=CachedSampleAvg()) where A = new{kind, A, typeof(returns)}(app, γ, α, returns)
end

(learner::MonteCarloLearner)(obs) = learner.approximator(get_state(obs))
(learner::MonteCarloLearner)(obs, a) = learner.approximator(get_state(s), a)

function update!(learner::MonteCarloLearner{:FirstVisit, <:AbstractVApproximator}, transitions)
    states, rewards = transitions
    V, γ, α, Returns, G, T =  learner.approximator, learner.γ, learner.α, learner.returns, 0., length(states)
    seen_states = countmap(states)

    for t in T:-1:1
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

function update!(learner::MonteCarloLearner{:EveryVisit, <:AbstractVApproximator}, transitions)
    states, rewards = transitions
    α, γ, V, Returns, G = learner.α, learner.γ, learner.approximator, learner.returns, 0.
    for (s, r) in Iterators.reverse(zip(states, rewards))
        G = γ * G + r
        update!(V, s => α * (Returns(s, G) - V(s)))
    end
end

function update!(learner::MonteCarloLearner{:FirstVisit,<:AbstractQApproximator}, transitions)
    states, actions, rewards = transitions
    α, γ, Q, Returns, G, T = learner.α, learner.γ, learner.approximator, learner.returns, 0., length(states)
    seen_pairs = countmap(zip(states, actions))

    for t in T:-1:1
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

function update!(learner::MonteCarloLearner{:EveryVisit, <:AbstractQApproximator}, transitions)
    states, actions, rewards = transitions
    α, γ, Q, Returns, G = learner.α, learner.γ, learner.approximator, learner.returns, 0.
    for (s, a, r) in Iterators.reverse(zip(states, actions, rewards))
        G = γ * G + r
        update!(Q, (s, a) => α * (Returns((s, a), G) - Q(s, a)))
    end
end