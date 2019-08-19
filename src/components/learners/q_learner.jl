export QLearner, update!

mutable struct QLearner{Tq<:QApproximator, Tf, Tl} <: AbstractLearner{Tq}
    approximator::Tq
    loss_fun::Tf
    γ::Float32
    loss::Tl
end

QLearner(Q, loss_fun, loss;γ=0.99f0) = QLearner(Q, loss_fun, γ, loss)

function update!(learner::QLearner{<:NeuralNetworkQ}, consecutive_batch)
    Q, γ, loss_fun = learner.approximator, learner.γ, learner.loss_fun
    n_step = size(consecutive_batch.states, ndims(consecutive_batch.states)-1)
    states, actions, rewards, terminals, next_states = extract_SARTS(consecutive_batch, γ)

    q = batch_estimate(Q, states, actions)
    q′ = dropdims(maximum(Q(next_states); dims=1), dims=1)
    G = rewards .+ γ^n_step .* (1 .- terminals) .* q′
    loss = loss_fun(G, q)
    learner.loss = loss
    if loss isa Tracker.TrackedReal{Float32}
        update!(Q, loss)
    else
        update!(Q, loss.loss)
    end
end