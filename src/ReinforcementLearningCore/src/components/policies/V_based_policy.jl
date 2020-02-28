export VBasedPolicy

"""
    VBasedPolicy(;kwargs...)

# Key words & Fields

- `learner`::[`AbstractLearner`](@ref), learn how to estimate state values.
- `mapping`, a customized function `(obs, learner) -> action_values`
- `explorer`::[`AbstractExplorer`](@ref), decide which action to take based on action values.
"""
Base.@kwdef struct VBasedPolicy{L<:AbstractLearner,M,E<:AbstractExplorer} <: AbstractPolicy
    learner::L
    mapping::M
    explorer::E = GreedyExplorer()
end

(p::VBasedPolicy)(obs, ::MinimalActionSet) = p.mapping(obs, p.learner) |> p.explorer

function (p::VBasedPolicy)(obs, ::FullActionSet)
    action_values = p.mapping(obs, p.learner)
    p.explorer(action_values, get_legal_actions_mask(obs))
end

function RLBase.get_prob(p::VBasedPolicy, obs, ::MinimalActionSet)
    get_prob(p.explorer, p.mapping(obs, p.learner))
end

function RLBase.get_prob(p::VBasedPolicy, obs, ::FullActionSet)
    action_values = p.mapping(obs, p.learner)
    get_prob(p.explorer, action_values, get_legal_actions_mask(obs))
end

RLBase.update!(p::VBasedPolicy, experience) = update!(p.learner, experience)

function RLBase.update!(p::VBasedPolicy, t::AbstractTrajectory)
    experience = extract_experience(t, p)
    isnothing(experience) || update!(p, experience)
end

RLBase.extract_experience(trajectory::AbstractTrajectory, p::VBasedPolicy) =
    extract_experience(trajectory, p.learner)
