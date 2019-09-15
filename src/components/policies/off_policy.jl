export OffPolicy

Base.@kwdef struct OffPolicy{P,B} <: AbstractPolicy
    π_target::P
    π_behavior::B
end

(π::OffPolicy)(obs::Observation) = π.π_behavior(obs)

function update!(
    π::OffPolicy{<:VBasedPolicy{<:MonteCarloLearner{T,A,R,S}}},
    transitions,
) where {T,A,R,S<:Union{OrdinaryImportanceSampling,WeightedImportanceSampling}}
    # ??? define a `get_batch_prob` function for efficiency
    weights = [get_prob(π.π_target, s, a) / get_prob(π.π_behavior, s, a) for (s, a) in zip(
        transitions.states,
        transitions.actions,
    )]  # TODO: implement iterate interface for (SubArray of) CircularArrayBuffer
    update!(π.π_target, transitions, weights)
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