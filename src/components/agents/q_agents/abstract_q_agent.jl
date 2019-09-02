export AbstractQAgent

abstract type AbstractQAgent <: AbstractAgent end

function (agent::AbstractQAgent)(mode::AbstractRuntimeMode, obs::Observation)
    obs |> get_state |> learner(agent) |> selector(mode, agent)
end

selector(agent::AbstractQAgent) = selector(mode(agent), agent)
selector(::TrainingMode, agent::AbstractQAgent) = agent.training_selector
selector(::EvaluatingMode, agent::AbstractQAgent) = agent.evaluating_selector
