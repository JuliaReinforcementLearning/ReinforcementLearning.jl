export QBasedPolicy

"""
    QBasedPolicy(learner::Q, explorer::S) -> QBasedPolicy{Q, S}
Use a Q-`learner` to generate the estimations of actions and use `explorer` to get the action.
"""
Base.@kwdef struct QBasedPolicy{
    Q<:AbstractLearner,
    E<:AbstractExplorer,
} <: AbstractPolicy
    learner::Q
    explorer::E
end

(π::QBasedPolicy)(obs, ::MinimalActionSet) = obs |> π.learner |> π.explorer
(π::QBasedPolicy)(obs, ::FullActionSet) = π.explorer(π.learner(obs), get_legal_actions_mask(obs))

function RLBase.update!(p::QBasedPolicy, t::AbstractTrajectory)
    experience = extract_experience(t, p)
    isnothing(experience) || update!(p.learner, experience)
end

RLBase.update!(p::QBasedPolicy, m::AbstractEnvironmentModel, t::AbstractTrajectory, n::Int) = update!(p.learner, m, t, n)

RLBase.extract_experience(trajectory::AbstractTrajectory, p::QBasedPolicy) = extract_experience(trajectory, p.learner)
RLBase.get_prob(p::QBasedPolicy, obs, ::MinimalActionSet) = get_prob(p.explorer, p.learner(obs))
RLBase.get_prob(p::QBasedPolicy, obs, ::FullActionSet) = get_prob(p.explorer, p.learner(obs), get_legal_actions_mask(obs))

RLBase.get_priority(p::QBasedPolicy, experience) = get_priority(p.learner, experience)