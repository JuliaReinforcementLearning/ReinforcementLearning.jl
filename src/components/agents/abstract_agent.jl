export AbstractAgent

abstract type AbstractAgent end

role(agent::AbstractAgent) = agent.role

learner(agent::AbstractAgent) = agent.learner

buffer(agent::AbstractAgent) = agent.buffer