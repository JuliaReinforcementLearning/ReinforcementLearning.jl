export DoubleQLearner, update!

using Flux

mutable struct DoubleQLearner{Tq<:QApproximator, Tf, Tl} <: AbstractLearner{Tq}
    approximator::Tq
    target_approximator::Tq
    loss_fun::Tf
    γ::Float32
    loss::Tl
    update_freq::Int
    update_step::Int
end

function DoubleQLearner(Q, target_Q, loss_fun, loss;γ=0.99f0, update_freq=100, update_step=0)
    copyto!(target_Q, Q)
    DoubleQLearner(Q, target_Q, loss_fun, γ, loss, update_freq, update_step)
end

function update!(learner::DoubleQLearner{<:NeuralNetworkQ}, consecutive_batch)
    Q, Qₜ, γ, loss_fun = learner.approximator, learner.target_approximator, learner.γ, learner.loss_fun
    n_step = size(consecutive_batch.states, ndims(consecutive_batch.states)-1)
    states, actions, rewards, terminals, next_states = extract_SARTS(consecutive_batch, γ)

    q = batch_estimate(Q, states, actions)
    q′ = dropdims(maximum(Qₜ(next_states); dims=1), dims=1)
    G = rewards .+ γ^n_step .* (1 .- terminals) .* q′
    loss = loss_fun(G, q)
    learner.loss = loss
    if loss isa Tracker.TrackedReal{Float32}
        update!(Q, loss)
    else
        update!(Q, loss.loss)
    end

    learner.update_step += 1
    if learner.update_step % learner.update_freq == 0
        copyto!(Qₜ, Q)
    end
end