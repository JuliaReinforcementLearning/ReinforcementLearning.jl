export DoubleQLearner, update!

using Flux
using StatsBase

mutable struct DoubleQLearner{Tq<:QApproximator, Tf, Tl} <: AbstractLearner{Tq}
    approximator::Tq
    target_approximator::Tq
    loss_fun::Tf
    γ::Float32
    loss::Tl
    update_freq::Int
    target_update_freq::Int
    update_step::Int
end

function DoubleQLearner(Q, target_Q, loss_fun;γ=0.99f0, update_freq=1, target_update_freq=100, update_step=0, loss=0.f0)
    copyto!(target_Q, Q)
    DoubleQLearner(Q, target_Q, loss_fun, γ, loss, update_freq, target_update_freq, update_step)
end

function update!(learner::DoubleQLearner{<:NeuralNetworkQ}, consecutive_batch)
    learner.update_step += 1
    learner.update_step % learner.update_freq == 0 || return nothing

    Q, Qₜ, γ, loss_fun = learner.approximator, learner.target_approximator, learner.γ, learner.loss_fun
    n_step = size(consecutive_batch.states, ndims(consecutive_batch.states)-1)
    states, actions, rewards, terminals, next_states = extract_SARTS(consecutive_batch, γ)

    q = batch_estimate(Q, states, actions)
    q′ = dropdims(maximum(Qₜ(next_states); dims=1), dims=1)
    G = rewards .+ γ^n_step .* (1 .- terminals) .* q′

    batch_losses = loss_fun(G, q)
    priorities = (batch_losses .+ 1f-10).data
    loss = mean(batch_losses)
    learner.loss = loss.data
    update!(Q, loss)

    if learner.update_step % learner.target_update_freq == 0
        copyto!(Qₜ, Q)
    end

    priorities
end