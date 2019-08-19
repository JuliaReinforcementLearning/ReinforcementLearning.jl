export QLearner, update!

mutable struct QLearner{Tq<:QApproximator, Tf, Tl} <: AbstractLearner{Tq}
    approximator::Tq
    loss_fun::Tf
    γ::Float32
    loss::Tl
end

QLearner(Q, loss_fun, loss;γ=0.99f0) = QLearner(Q, loss_fun, γ, loss)

function extract(learner, batch)
    γ = learner.γ
    n_step, batch_size = size(batch.terminals)
    states = selectdim(batch.states, ndims(batch.states)-1, 1)
    actions = selectdim(batch.actions, ndims(batch.actions)-1, 1)
    next_states = selectdim(batch.next_states, ndims(batch.next_states)-1, n_step)

    rewards, terminals = zeros(Float32, batch_size), fill(false, batch_size)

    # make sure that we only consider experiences in current episode
    for i in 1:batch_size
        t = findfirst(view(batch.terminals, :, i))

        if isnothing(t)
            terminals[i] = false
            rewards[i] = discount_rewards_reduced(view(batch.rewards[:, i]), γ)
        else
            terminals[i] = true
            rewards[i] = discount_rewards_reduced(view(batch.rewards[1:t, i]), γ)
        end
    end

    states, actions, rewards, terminals, next_states
end

function update!(learner::QLearner{<:NeuralNetworkQ}, consecutive_batch)
    Q, γ, loss_fun = learner.approximator, learner.γ, learner.loss_fun
    n_step = size(consecutive_batch.states, ndims(consecutive_batch.states)-1)
    states, actions, rewards, terminals, next_states = extract(learner, consecutive_batch)

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