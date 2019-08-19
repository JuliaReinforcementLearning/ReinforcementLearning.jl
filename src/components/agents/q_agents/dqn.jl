export DQN

mutable struct DQN{Tl, Tb<:AbstractTurnBuffer, Ts<:AbstractDiscreteActionSelector} <: AbstractQAgent
    role::String
    learner::Tl
    buffer::Tb
    selector::Ts
    batch_size::Int
    update_horizon::Int  # starts with 1
    γ::Float64
    act_step::Int
    min_replay_history::Int
    default_priority::Float64  # ignored if `buffer` doesn't contain `priority`
end

function selector(agent::DQN{Tl, Tb, <:EpsilonGreedySelector}) where {Tl, Tb}
    x -> begin
        a = agent.selector(x;step=agent.act_step)
        agent.act_step += 1
        a
    end
end

DQN(learner, buffer, selector; batch_size=32, update_horizon=1, γ=0.99, role="DEFAULT", act_step=0, min_replay_history=32, default_priority=100.) = DQN(role, learner, buffer, selector, batch_size, update_horizon, γ, act_step, min_replay_history, default_priority)

function update!(agent::DQN{Tl, Tb}, experience::Pair) where {Tl, Tb<:CircularTurnBuffer{RTSA}}
    push!(buffer(agent), experience)
    if length(buffer(agent)) > agent.min_replay_history
        inds, batch = sample(buffer(agent); batch_size=agent.batch_size, n_step=agent.update_horizon)
        update!(agent.learner, batch)
    end
end

function update!(agent::DQN{Tl, Tb}, experience::Pair) where {Tl, Tb<:CircularTurnBuffer{PRTSA}}
    push!(priority(buffer(agent)), agent.default_priority)
    push!(buffer(agent), experience)
    if length(buffer(agent)) > agent.min_replay_history
        inds, batch = sample(buffer(agent); batch_size=agent.batch_size, n_step=agent.update_horizon)
        update!(agent.learner, batch)

        # agent.learner.loss_fun is expected to return `loss` and `batch_losses`
        priorities = agent.learner.loss.batch_losses .+ 1f-10
        priority(buffer(agent))[inds] .= priorities.data
    end
end