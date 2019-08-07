export QLearner, update!

struct QLearner{Tq<:QApproximator, Tf} <: AbstractLearner{Tq}
    Q::Tq
    γ::Float64
    n::Int
    loss_fun::Tf
end

function update!(learner::QLearner{<:NeuralNetworkQ}, batch::NamedTuple{(:states, :actions, :rewards, :terminals, :next_states)})
    Q, γ, loss_fun = learner.Q, learner.γ, learner.loss_fun

    q = Q(batch.states, batch.actions)
    q′ = maximum(Q(batch.next_states); dims=1)
    G = batch.rewards .+ γ .* (1 .- batch.terminals) .* q′
    loss = loss_fun(G, q)
    update!(Q, loss)
end