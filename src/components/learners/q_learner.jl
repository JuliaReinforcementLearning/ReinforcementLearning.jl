export QLearner, update!

mutable struct QLearner{Tq<:QApproximator, Tf} <: AbstractLearner{Tq}
    approximator::Tq
    loss_fun::Tf
    γ::Float64
    loss::Float32
end

QLearner(Q, loss_fun;γ=0.99) = QLearner(Q, loss_fun, γ, 0f0)

function update!(learner::QLearner{<:NeuralNetworkQ}, batch::NamedTuple{(:states, :actions, :rewards, :terminals, :next_states, :next_actions)})
    Q, γ, loss_fun = learner.approximator, learner.γ, learner.loss_fun
    n_step, batch_size = size(batch.terminals)

    states = selectdim(batch.states, ndims(batch.states)-1, 1)
    actions = selectdim(batch.actions, ndims(batch.actions)-1, 1)
    next_states = selectdim(batch.next_states, ndims(batch.next_states)-1, n_step)

    rewards, terminals = zeros(Float32, batch_size), fill(false, batch_size)

    # make sure that we only consider experiences in current episode
    for i in 1:n_step
        t = findfirst(view(batch.terminals, :, i))

        if isnothing(t)
            terminals[i] = false
            rewards[i] = discount_rewards_reduced(view(batch.rewards[:, i]), γ)
        else
            terminals[i] = true
            rewards[i] = discount_rewards_reduced(view(batch.rewards[1:t, i]), γ)
        end
    end

    q = batch_estimate(Q, states, actions)
    q′ = dropdims(maximum(Q(next_states); dims=1), dims=1)
    G = rewards .+ γ^(n_step-1) .* (1 .- terminals) .* q′
    loss = loss_fun(G, q)
    learner.loss = loss.data
    update!(Q, loss)
end