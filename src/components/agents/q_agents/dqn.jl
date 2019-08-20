export DQN

using Flux

mutable struct DQN{Tl, Tb<:AbstractTurnBuffer, Ts<:AbstractDiscreteActionSelector} <: AbstractQAgent
    role::String
    learner::Tl
    buffer::Tb
    selector::Ts
    batch_size::Int
    update_horizon::Int  # starts with 1
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

DQN(learner, buffer, selector; batch_size=32, update_horizon=1, role="DEFAULT", act_step=0, min_replay_history=32, default_priority=100.) = DQN(role, learner, buffer, selector, batch_size, update_horizon, act_step, min_replay_history, default_priority)

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
        priorities = update!(agent.learner, batch)
        isnothing(priorities) || (priority(buffer(agent))[inds] .= priorities)
    end
end

function (agent::DQN{<:RainbowLearner})(obs::EnvObservation)
    logits = obs |> state |> learner(agent)
    q = agent.learner.support .* softmax(reshape(logits, :, agent.learner.n_actions))
    # probs = vec(sum(q, dims=1)) .+ legal_action
    probs = vec(sum(q, dims=1))
    probs |> selector(agent)
end
