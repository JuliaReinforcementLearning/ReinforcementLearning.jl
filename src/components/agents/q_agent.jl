abstract type AbstractQAgent end

selector(agent::AbstractQAgent) = agent.selector

(agent::AbstractQAgent)(s) = s |> learner(agent) |> selector(agent)

struct DQN{Tl<:QLearner{<:NeuralNetworkQ}, Tb<:AbstractTurnBuffer, Ts<:AbstractDiscreteActionSelector} <: AbstractQAgent
    role::String
    learner::Tl
    buffer::Tb
    selector::Ts
    batch_size::Int
    update_horizon::Int  # starts with 1
    λ::Float64
end

function update!(agent::DQN)
    batch = sample(buffer(agent); batch_size=agent.batch_size, n_step=agent.update_horizon)
    update!(agent.learner, batch)
end