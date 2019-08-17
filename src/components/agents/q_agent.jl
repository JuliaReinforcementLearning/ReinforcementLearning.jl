export AbstractQAgent, DQN

abstract type AbstractQAgent <: AbstractAgent end

selector(agent::AbstractQAgent) = agent.selector

function (agent::AbstractQAgent)(obs::EnvObservation)
    obs |> state |> learner(agent) |> selector(agent)
end

mutable struct DQN{Tl<:QLearner{<:NeuralNetworkQ}, Tb<:AbstractTurnBuffer, Ts<:AbstractDiscreteActionSelector} <: AbstractQAgent
    role::String
    learner::Tl
    buffer::Tb
    selector::Ts
    batch_size::Int
    update_horizon::Int  # starts with 1
    γ::Float64
    act_step::Int
    min_replay_history::Int
end

function selector(agent::DQN{Tl, Tb, <:EpsilonGreedySelector}) where {Tl, Tb}
    x -> begin
        a = agent.selector(x;step=agent.act_step)
        agent.act_step += 1
        a
    end
end

DQN(learner, buffer, selector; batch_size=32, update_horizon=1, γ=0.99, role="DEFAULT", act_step=0, min_replay_history=32) = DQN(role, learner, buffer, selector, batch_size, update_horizon, γ, act_step, min_replay_history)

function update!(agent::DQN, experience::Pair)
    push!(buffer(agent), experience)
    if length(buffer(agent)) > agent.min_replay_history
        batch = sample(buffer(agent); batch_size=agent.batch_size, n_step=agent.update_horizon)
        update!(agent.learner, batch)
    end
end