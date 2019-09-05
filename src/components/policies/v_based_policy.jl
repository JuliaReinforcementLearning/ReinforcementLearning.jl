export VBasedPolicy

struct VBasedPolicy{V, F} <: AbstractPolicy
    learner::V
    f::F
end

(π::VBasedPolicy)(s) = π.F(π.learner, s)
(π::VBasedPolicy)(s, a) = π.F(π.learner, s, a)

update!(π::VBasedPolicy, args...) = update!(π.learner, args...)

extract_transitions(buffer, π::VBasedPolicy) = extract_transitions(buffer, π.learner)

function extract_transitions(buffer::EpisodeTurnBuffer, ::MonteCarloLearner{T, A}) where {T, A<:AbstractVApproximator}
    if isfull(buffer)
        @views (state(buffer)[1:end-1], rewards(buffer)[2:end])
    else
        nothing
    end
end