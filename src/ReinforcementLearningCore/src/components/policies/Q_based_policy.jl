export QBasedPolicy

using MacroTools: @forward
using Flux
using Setfield

"""
    QBasedPolicy(;learner::Q, explorer::S)

Use a Q-`learner` to generate estimations of action values.
Then an `explorer` is applied on the estimations to select an action.
"""
Base.@kwdef struct QBasedPolicy{Q<:AbstractLearner,E<:AbstractExplorer} <: AbstractPolicy
    learner::Q
    explorer::E
end

Flux.functor(x::QBasedPolicy) = (learner = x.learner,), y -> @set x.learner = y.learner

(π::QBasedPolicy)(obs) = π(obs, ActionStyle(obs))
(π::QBasedPolicy)(obs, ::MinimalActionSet) = obs |> π.learner |> π.explorer
(π::QBasedPolicy)(obs, ::FullActionSet) =
    π.explorer(π.learner(obs), get_legal_actions_mask(obs))

RLBase.get_prob(p::QBasedPolicy, obs) = get_prob(p, obs, ActionStyle(obs))
RLBase.get_prob(p::QBasedPolicy, obs, ::MinimalActionSet) =
    get_prob(p.explorer, p.learner(obs))
RLBase.get_prob(p::QBasedPolicy, obs, ::FullActionSet) =
    get_prob(p.explorer, p.learner(obs), get_legal_actions_mask(obs))

@forward QBasedPolicy.learner RLBase.get_priority, RLBase.update!

function Flux.testmode!(p::QBasedPolicy, mode = true)
    testmode!(p.learner, mode)
    testmode!(p.explorer, mode)
end
