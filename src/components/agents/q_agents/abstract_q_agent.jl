export AbstractQAgent

abstract type AbstractQAgent <: AbstractAgent end

selector(agent::AbstractQAgent) = agent.selector

function (agent::AbstractQAgent)(obs::Observation)
    obs |> get_state |> learner(agent) |> selector(agent)
end
