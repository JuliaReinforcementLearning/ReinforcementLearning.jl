export BasicDQNLearner

using StatsBase: mean

"""
https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
"""
Base.@kwdef mutable struct BasicDQNLearner{Tq<:AbstractQApproximator,Tf} <: AbstractLearner
    approximator::Tq
    loss_fun::Tf
    γ::Float32 = 0.99f0
    batch_size::Int = 32
    update_horizon::Int = 1
    min_replay_history::Int = 32
    loss::Float32 = 0.f0  # used to record
end

function update!(learner::BasicDQNLearner{<:NeuralNetworkQ}, batch)
    Q, γ, loss_fun, update_horizon = learner.approximator,
        learner.γ,
        learner.loss_fun,
        learner.update_horizon
    states, actions, rewards, terminals, next_states = batch

    q = batch_estimate(Q, states, actions)
    q′ = dropdims(maximum(Q(next_states); dims = 1), dims = 1)
    G = rewards .+ γ^update_horizon .* (1 .- terminals) .* q′

    batch_losses = loss_fun(to_device(Q, G), q)
    loss = mean(batch_losses)
    learner.loss = loss.data
    update!(Q, loss)
end

function extract_transitions(buffer::CircularTurnBuffer{RTSA}, learner::BasicDQNLearner)
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