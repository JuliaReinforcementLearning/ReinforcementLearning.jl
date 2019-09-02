export MonteCarloAgent, update!

mutable struct MonteCarloAgent{L<:MonteCarloLearner, P, B, R, M<:AbstractRuntimeMode} <: AbstractAgent
    role::R
    mode::M
    learner::L
    π::P
    buffer::B
end

(agent::MonteCarloAgent)(obs::Observation) = agent.π(obs)

function update!(::TrainingMode, agent::MonteCarloAgent{L, P, <:EpisodeTurnBuffer}, experience::Pair) where {L, P}
    buf = buffer(agent)
    push!(buf, experience)
    if isfull(buf)
        @views update!(agent.learner, state(buf)[1:end-1], reward(buf)[1:end])
    end
end