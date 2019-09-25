export VBasedPolicy

Base.@kwdef struct VBasedPolicy{V,F} <: AbstractPolicy
    learner::V
    f::F
end

(π::VBasedPolicy)(obs) = π.f(obs)

get_prob(π::VBasedPolicy, s, a) = get_prob(π.f, s, a)

extract_transitions(buffer, π::VBasedPolicy) = extract_transitions(buffer, π.learner)

function extract_transitions(
    buffer::EpisodeTurnBuffer,
    ::MonteCarloLearner{T,A},
) where {T,A<:AbstractVApproximator}
    if isfull(buffer)
        @views (states = state(buffer)[1:end-1], rewards = reward(buffer)[2:end])
    else
        nothing
    end
end