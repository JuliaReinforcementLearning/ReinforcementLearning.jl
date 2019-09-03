export AbstractQAgent

abstract type AbstractQAgent <: AbstractAgent end

selector(agent::AbstractQAgent) = agent.selector
inc_act_step(agent::AbstractQAgent) = agent.act_step += 1

function (agent::AbstractQAgent)(obs::Observation)
    inc_act_step(agent)
    obs |> get_state |> learner(agent) |> selector(agent)
end