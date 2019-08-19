export AbstractQAgent

abstract type AbstractQAgent <: AbstractAgent end

selector(agent::AbstractQAgent) = agent.selector

function (agent::AbstractQAgent)(obs::EnvObservation)
    obs |> state |> learner(agent) |> selector(agent)
end
