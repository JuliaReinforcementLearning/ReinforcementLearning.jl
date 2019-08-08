export QLearner, update!

mutable struct QLearner{Tq<:QApproximator, Tf} <: AbstractLearner{Tq}
    approximator::Tq
    loss_fun::Tf
    γ::Float64
    loss::Float32
end

QLearner(Q, loss_fun;γ=0.99) = QLearner(Q, loss_fun, γ, 0f0)

function update!(learner::QLearner{<:NeuralNetworkQ}, batch::NamedTuple{(:states, :actions, :rewards, :terminals, :next_states)})
    Q, γ, loss_fun = learner.approximator, learner.γ, learner.loss_fun

    q = batch_estimate(Q, batch.states, batch.actions)
    q′ = dropdims(maximum(Q(batch.next_states); dims=1), dims=1)
    G = batch.rewards .+ γ .* (1 .- batch.terminals) .* q′
    loss = loss_fun(G, q)
    learner.loss = loss.data
    update!(Q, loss)
end