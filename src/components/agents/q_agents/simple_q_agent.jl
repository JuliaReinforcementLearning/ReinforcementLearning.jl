export SimpleQAgent, update!

mutable struct SimpleQAgent{L, S, B} <: AbstractQAgent
    learner::L
    selector::S
    buffer::B
end

SimpleQAgent(;learner, selector, buffer) = SimpleQAgent(learner, selector, buffer)

function extract_experience(buffer::EpisodeTurnBuffer, learner::TDLearner{<:AbstractQApproximator, :SARSA})
    @views (
        state(buffer)[1:end-1],
        action(buffer)[1:end-1],
        reward(buffer)[2:end],
        terminal(buffer)[2:end],
        state(buffer)[2:end],
        action(buffer)[2:end]
    )
end

function extract_experience(buffer::EpisodeTurnBuffer, learner::GradientBanditLearner)
    if length(buffer) > 0
        state(buffer)[end-1], action(buffer)[end-1], reward(buffer)[end]
    else
        nothing
    end
end

function update!(agent::SimpleQAgent{L, S, B}, experience::Pair) where {L<:TDLearner, S, B<:EpisodeTurnBuffer}
    push!(buffer(agent), experience)
    update!(agent.learner, extract_experience(agent.buffer, agent.learner)...)
end

function update!(agent::SimpleQAgent, experience::Pair)
    push!(buffer(agent), experience)
    exp = extract_experience(buffer(agent), agent.learner)
    if !isnothing(exp)
        update!(agent.learner, exp...)
    end
end