export DQNLearner, update!

using Flux
using Zygote
using StatsBase

"""
    DQNLearner(;kwargs...)

See paper: [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)

# Keywords

- `approximator`::[`AbstractQApproximator`](@ref): used to get Q-values of a state.
- `target_approximator`::[`AbstractQApproximator`](@ref): similar to `approximator`, but used to estimate the target (the next state).
- `loss_fun`: the loss function.
- `γ::Float32=0.99f0`: discount rate.
- `batch_size::Int=32`
- `update_horizon::Int=1`: length of update ('n' in n-step update).
- `min_replay_history::Int=32`: number of transitions that should be experienced before updating the `approximator`.
- `update_freq::Int=4`: the frequency of updating the `approximator`.
- `target_update_freq::Int=100`: the frequency of syncing `target_approximator`.
- `stack_size::Union{Int, Nothing}=4`: use the recent `stack_size` frames to form a stacked state.

"""
Base.@kwdef mutable struct DQNLearner{Tq<:AbstractQApproximator, Tt<:AbstractQApproximator, Tf, Ts<:Union{Int, Nothing}} <: AbstractLearner
    approximator::Tq
    target_approximator::Tt
    loss_fun::Tf
    stack_size::Ts = 4
    γ::Float32 = 0.99f0
    batch_size::Int = 32
    update_horizon::Int = 1
    min_replay_history::Int = 32
    update_freq::Int = 1
    target_update_freq::Int = 100
    update_step::Int = 0

    function DQNLearner(approximator::Tq, target_approximator::Tt, loss_fun::Tf, stack_size::Ts, args...) where {Tq, Tt, Tf, Ts}
        copyto!(approximator, target_approximator)
        new{Tq, Tt, Tf, Ts}(approximator, target_approximator, loss_fun, stack_size, args...)
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

    gs = gradient(Q) do 
        q = batch_estimate(Q, states)[actions]
        q′ = dropdims(maximum(batch_estimate(Qₜ, next_states); dims = 1), dims = 1)
        G = rewards .+ γ^update_horizon .* (1 .- terminals) .* q′

        batch_losses = loss_fun(G, q)
        mean(batch_losses)
    end

    update!(Q, gs)

    if learner.update_step % learner.target_update_freq == 0
        copyto!(Qₜ, Q)
    end
end

function extract_transitions(buffer::CircularTurnBuffer{RTSA}, learner::DQNLearner)
    if length(buffer) > learner.min_replay_history
        inds, consecutive_batch = sample(buffer, learner.batch_size, learner.update_horizon, learner.stack_size)
        extract_SARTS(consecutive_batch, learner.γ)
    else
        nothing
    end
end