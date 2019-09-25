export OffPolicy

Base.@kwdef struct OffPolicy{P,B} <: AbstractPolicy
    π_target::P
    π_behavior::B
end

function update!(π::OffPolicy, buffer::AbstractTurnBuffer; kw...)
    transitions = extract_transitions(buffer, π)
    if !isnothing(transitions)
        update!(π, transitions; kw...)
    end
end

learner(π::OffPolicy) = learner(π.π_target)

(π::OffPolicy)(obs::Observation) = π.π_behavior(obs)

function update!(π::OffPolicy{<:VBasedPolicy}, transitions::NamedTuple)
    # ??? define a `get_batch_prob` function for efficiency
    weights = [get_prob(π.π_target, s, a) / get_prob(π.π_behavior, s, a) for (s, a) in zip(
        transitions.states,
        transitions.actions,
    )]  # TODO: implement iterate interface for (SubArray of) CircularArrayBuffer
    update!(learner(π.π_target), transitions, weights)
end

function extract_transitions(
    buffer,
    π::OffPolicy{<:VBasedPolicy{<:MonteCarloLearner{T,A,R,S}}},
) where {T,A,R,S<:Union{OrdinaryImportanceSampling,WeightedImportanceSampling}}
    if isfull(buffer)
        @views (
            states = state(buffer)[1:end-1],
            actions = action(buffer)[1:end-1],
            rewards = reward(buffer)[2:end],
        )
    else
        nothing
    end
end

function extract_transitions(
    buffer::EpisodeTurnBuffer,
    π::OffPolicy{<:VBasedPolicy{<:TDLearner{<:AbstractVApproximator,:SRS}}},
)
    transitions = extract_transitions(buffer, π.π_target.learner)
    if isnothing(transitions)
        nothing
    else
        merge(
            transitions,
            (actions = action(buffer)[max(1, end - π.π_target.learner.n - 1):end-1],),
        )
    end
end