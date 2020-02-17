export OffPolicy

"""
    OffPolicy(π_target::P, π_behavior::B) -> OffPolicy{P,B}
"""
Base.@kwdef struct OffPolicy{P,B} <: AbstractPolicy
    π_target::P
    π_behavior::B
end

(π::OffPolicy)(obs) = π.π_behavior(obs)

function RLBase.update!(π::OffPolicy, t::AbstractTrajectory)
    experience = extract_experience(t, π)
    isnothing(experience) || update!(π, experience)
end

function RLBase.update!(π::OffPolicy{<:VBasedPolicy}, transitions::NamedTuple)
    # ??? define a `get_batch_prob` function for efficiency
    weights = [
        get_prob(π.π_target, (state = s,), a) / get_prob(π.π_behavior, (state = s,), a)
        for (s, a) in zip(transitions.states, transitions.actions)
    ]  # TODO: implement iterate interface for (SubArray of) CircularArrayBuffer
    experience = merge(transitions, (weights = weights,))
    update!(π.π_target, experience)
end

RLBase.extract_experience(t::AbstractTrajectory, π::OffPolicy) =
    extract_experience(t, π.π_target)
