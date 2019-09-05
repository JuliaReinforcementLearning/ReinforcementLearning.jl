export PrioritizedDQNLearner

Base.@kwdef mutable struct PrioritizedDQNLearner{Tq<:AbstractQApproximator, Tf, Tl} <: AbstractLearner
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
    loss::Tl = 0.f0  # used to record
    default_priority::Float64
end

function update!(learner::PrioritizedDQNLearner{<:NeuralNetworkQ}, indexed_batch)
    learner.update_step += 1
    learner.update_step % learner.update_freq == 0 || return nothing

    inds, batch = indexed_batch
    Q, Qₜ, γ, loss_fun, update_horizon = learner.approximator, learner.target_approximator, learner.γ, learner.loss_fun, learner.update_horizon
    states, actions, rewards, terminals, next_states = batch

    q = batch_estimate(Q, states, actions)
    q′ = dropdims(maximum(Qₜ(next_states); dims=1), dims=1)
    G = rewards .+ γ^update_horizon .* (1 .- terminals) .* q′

    batch_losses = loss_fun(G, q)
    priorities = (batch_losses .+ 1f-10).data
    loss = mean(batch_losses)
    learner.loss = loss.data
    update!(Q, loss)

    if learner.update_step % learner.target_update_freq == 0
        copyto!(Qₜ, Q)
    end

    inds, priorities
end