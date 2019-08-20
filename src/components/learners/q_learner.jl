export QLearner, update!

mutable struct QLearner{Tq<:QApproximator, Tf, Tl} <: AbstractLearner{Tq}
    approximator::Tq
    loss_fun::Tf
    γ::Float32
    loss::Tl
end

QLearner(Q, loss_fun;γ=0.99f0, loss=0.f0) = QLearner(Q, loss_fun, γ, loss)

function update!(learner::QLearner{<:NeuralNetworkQ}, consecutive_batch)
    Q, γ, loss_fun = learner.approximator, learner.γ, learner.loss_fun
    n_step = size(consecutive_batch.states, ndims(consecutive_batch.states)-1)
    states, actions, rewards, terminals, next_states = extract_SARTS(consecutive_batch, γ)

    q = batch_estimate(Q, states, actions)
    q′ = dropdims(maximum(Q(next_states); dims=1), dims=1)
    G = rewards .+ γ^n_step .* (1 .- terminals) .* q′

    batch_losses = loss_fun(G, q)
    priorities = (batch_losses .+ 1f-10).data
    loss = mean(batch_losses)
    learner.loss = loss.data
    update!(Q, loss)

    priorities
end