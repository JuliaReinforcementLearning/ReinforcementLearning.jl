export BasicDQNLearner

using StatsBase: mean
using Zygote

"""
    BasicDQNLearner(;kwargs...)

See paper: [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)

This is the very basic implementation of DQN. Compared to the traditional Q learning, the only difference is that,
in the updating step it uses a batch of transitions sampled from an experience buffer instead of current transition.
And the `approximator` is usually a [`NeuralNetworkQ`](@ref).

You can start from this implementation to understand how everything is organized and how to write your own customized algorithm.

# Keywords

- `approximator`::[`AbstractQApproximator`](@ref): used to get Q-values of a state.
- `loss_fun`: the loss function to use. TODO: provide a default [`huber_loss`](@ref)?
- `γ::Float32=0.99f0`: discount rate.
- `batch_size::Int=32`
- `update_horizon::Int=1`: length of update ('n' in n-step update).
- `min_replay_history::Int=32`: number of transitions that should be experienced before updating the `approximator`.
"""
Base.@kwdef mutable struct BasicDQNLearner{Tq<:AbstractQApproximator,Tf} <: AbstractLearner
    approximator::Tq
    loss_fun::Tf
    γ::Float32 = 0.99f0
    batch_size::Int = 32
    update_horizon::Int = 1
    min_replay_history::Int = 32
end

function update!(learner::BasicDQNLearner{<:NeuralNetworkQ}, batch)
    Q, γ, loss_fun, update_horizon, batch_size = learner.approximator,
        learner.γ,
        learner.loss_fun,
        learner.update_horizon,
        learner.batch_size
    states, rewards, terminals, next_states = map(x->to_device(Q, x), (batch.states, batch.rewards, batch.terminals, batch.next_states))
    actions = CartesianIndex.(batch.actions, 1:batch_size) 

    gs = gradient(Q) do 
        q = batch_estimate(Q, states)[actions]
        q′ = vec(maximum(batch_estimate(Q, next_states); dims = 1))
        G = rewards .+ γ^update_horizon .* (1 .- terminals) .* q′

        batch_losses = loss_fun(G, q)
        mean(batch_losses)
    end

    update!(Q, gs)
end

function extract_transitions(buffer::CircularTurnBuffer{RTSA}, learner::BasicDQNLearner)
    if length(buffer) > learner.min_replay_history
        inds, consecutive_batch = sample(buffer, learner.batch_size, learner.update_horizon, nothing)
        extract_SARTS(consecutive_batch, learner.γ)
    else
        nothing
    end
end