export DQNLearner, update!

using Flux
using StatsBase

"""
    DQNLearner(;kwargs...)

See paper: [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)

# Keywords

TODO: need review
"""
Base.@kwdef mutable struct DQNLearner{Tq<:AbstractQApproximator,Tf} <: AbstractLearner
    approximator::Tq
    target_approximator::Tq
    loss_fun::Tf
    γ::Float32 = 0.99f0
    batch_size::Int = 32
    update_horizon::Int = 1
    min_replay_history::Int = 32
    update_freq::Int = 1
    target_update_freq::Int = 100
    update_step::Int = 0
    loss::Float32 = 0.f0

    function DQNLearner(approximator::Tq, target_approximator::Tq, loss_fun::Tf, args...) where {Tq, Tf}
        copyto!(approximator, target_approximator)
        new{Tq, Tf}(approximator, target_approximator, loss_fun, args...)
    end
end

function update!(learner::DQNLearner{<:NeuralNetworkQ}, batch)
    learner.update_step += 1
    learner.update_step % learner.update_freq == 0 || return nothing

    Q, Qₜ, γ, loss_fun, update_horizon, batch_size = learner.approximator,
        learner.target_approximator,
        learner.γ,
        learner.loss_fun,
        learner.update_horizon,
        learner.batch_size
    states, rewards, terminals, next_states = map(x->to_device(Q, x), (batch.states, batch.rewards, batch.terminals, batch.next_states))
    actions = CartesianIndex.(batch.actions, 1:batch_size) 

    loss, back = Flux.pullback(Q.params) do 
        q = batch_estimate(Q, states)[actions]
        q′ = dropdims(maximum(batch_estimate(Qₜ, next_states); dims = 1), dims = 1)
        G = rewards .+ γ^update_horizon .* (1 .- terminals) .* q′

        batch_losses = loss_fun(G, q)
        mean(batch_losses)
    end

    learner.loss = loss
    update!(Q, back(loss))

    if learner.update_step % learner.target_update_freq == 0
        copyto!(Qₜ, Q)
    end
end

function extract_transitions(buffer::CircularTurnBuffer{RTSA}, learner::DQNLearner)
    if length(buffer) > learner.min_replay_history
        inds, consecutive_batch = sample(
            buffer;
            batch_size = learner.batch_size,
            n_step = learner.update_horizon,
        )
        extract_SARTS(consecutive_batch, learner.γ)
    else
        nothing
    end
end