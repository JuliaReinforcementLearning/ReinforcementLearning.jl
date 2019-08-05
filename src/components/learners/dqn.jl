export DQN, update!

struct DQN{Tq<:QApproximator, Tf} <: AbstractLearner
    Q::Tq
    γ::Float64
    n::Int
    loss_fun::Tf
end

function update!(learner::DQN, states, actions, discount_rewards, terminals, next_states)
    Q, γ, loss_fun = learner.Q, learner.γ, learner.loss_fun

    q = Q(states, actions)
    q′ = maximum(Q(next_states); dims=1)
    G = discount_rewards .+ γ .* (1 .- terminals) .* q′
    loss = loss_fun(G, q)
    update!(Q, loss)
end