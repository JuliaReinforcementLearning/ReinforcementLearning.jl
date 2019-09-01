export MonteCarloAgent, update!

struct MonteCarloAgent{L<:MonteCarloLearner, P, B, R} <: AbstractAgent
    role::R
    learner::L
    π::P
    buffer::B
end

(agent::MonteCarloAgent)(obs::Observation) = agent.π(obs)

function update!(agent::MonteCarloAgent{L, P, <:EpisodeTurnBuffer}, experience::Pair) where {L, P}
    buf = buffer(agent)
    push!(buf, experience)
    if isfull(buf)
        @views update!(agent.learner, state(buf)[1:end-1], reward(buf)[1:end])
    end
    update!(agent, agent.π)
end

# not sure how to update the policy yet, provide the interface to let the user to override
update!(agent::MonteCarloAgent{L, P}, π::P) where {L, P} = nothing