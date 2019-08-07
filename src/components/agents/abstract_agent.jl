export AbstractAgent

abstract type AbstractAgent end

role(agent::AbstractAgent) = agent.role

learner(agent::AbstractAgent) = agent.learner

buffer(agent::AbstractAgent) = agent.buffer

function (agent::AbstractAgent)(obs::EnvObservation)
    s = state(obs)
    a = agent(s)
    update!(agent, obs => a)
end

function update!(agent::AbstractAgent, experience::Pair)
    obs, a = experience
    update!(buffer(agent); reward=reward(obs), terminal=terminal(obs), state=state(obs), action=a)
    update!(agent)
end