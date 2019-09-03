export AbstractAgent, update!

abstract type AbstractAgent end

abstract type AbstractRuntimeMode end
struct TrainingMode <:AbstractRuntimeMode end
struct EvaluatingMode <:AbstractRuntimeMode end
const TRAINING_MODE = TrainingMode()
const EVALUATING_MODE = EvaluatingMode()

role(agent::AbstractAgent) = agent.role

mode(agent::AbstractAgent) = agent.mode

learner(agent::AbstractAgent) = agent.learner

buffer(agent::AbstractAgent) = agent.buffer

update!(agent::AbstractAgent, experience) = update!(mode(agent), agent, experience)

update!(::EvaluatingMode, agent, experience) = nothing

(agent::AbstractAgent)(obs::Observation) = agent(mode(agent), obs)