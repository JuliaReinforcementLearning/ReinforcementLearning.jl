export VBasedPolicy

using MacroTools: @forward

"""
    VBasedPolicy(;learner, mapping, explorer=GreedyExplorer())

# Key words & Fields

- `learner`::[`AbstractLearner`](@ref), learn how to estimate state values.
- `mapping`, a customized function `(env, learner) -> action_values`
- `explorer`::[`AbstractExplorer`](@ref), decide which action to take based on action values.
"""
Base.@kwdef struct VBasedPolicy{L<:AbstractLearner,M,E<:AbstractExplorer} <: AbstractPolicy
    learner::L
    mapping::M
    explorer::E = GreedyExplorer()
end

(p::VBasedPolicy)(env) = p(env, ActionStyle(env))

(p::VBasedPolicy)(env, ::MinimalActionSet) = p.mapping(env, p.learner) |> p.explorer

function (p::VBasedPolicy)(env, ::FullActionSet)
    action_values = p.mapping(env, p.learner)
    p.explorer(action_values, get_legal_actions_mask(env))
end

RLBase.get_prob(p::VBasedPolicy, env, action::Integer) =
    get_prob(p, env, ActionStyle(env), action)

RLBase.get_prob(p::VBasedPolicy, env, ::MinimalActionSet) =
    get_prob(p.explorer, p.mapping(env, p.learner))
RLBase.get_prob(p::VBasedPolicy, env, ::MinimalActionSet, action) =
    get_prob(p.explorer, p.mapping(env, p.learner), action)

function RLBase.get_prob(p::VBasedPolicy, env, ::FullActionSet)
    action_values = p.mapping(env, p.learner)
    get_prob(p.explorer, action_values, get_legal_actions_mask(env))
end

function RLBase.get_prob(p::VBasedPolicy, env, ::FullActionSet, action)
    action_values = p.mapping(env, p.learner)
    get_prob(p.explorer, action_values, get_legal_actions_mask(env), action)
end

@forward VBasedPolicy.learner RLBase.get_priority, RLBase.update!
