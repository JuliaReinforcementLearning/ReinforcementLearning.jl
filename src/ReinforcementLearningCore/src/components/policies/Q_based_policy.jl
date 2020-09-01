export QBasedPolicy

using MacroTools: @forward
using Flux
using Setfield

"""
    QBasedPolicy(;learner::Q, explorer::S)

Use a Q-`learner` to generate estimations of action values.
Then an `explorer` is applied on the estimations to select an action.
"""
Base.@kwdef mutable struct QBasedPolicy{Q<:AbstractLearner,E<:AbstractExplorer} <:
                           AbstractPolicy
    learner::Q
    explorer::E
end

Flux.functor(x::QBasedPolicy) = (learner = x.learner,), y -> @set x.learner = y.learner

(π::QBasedPolicy)(env) = π(env, ActionStyle(env))
(π::QBasedPolicy)(env, ::MinimalActionSet) = get_actions(env)[envπ.learnerπ.explorer]
(π::QBasedPolicy)(env, ::FullActionSet) =
    get_actions(env)[π.explorer(π.learner(env), get_legal_actions_mask(env))]

RLBase.get_prob(p::QBasedPolicy, env) = get_prob(p, env, ActionStyle(env))
RLBase.get_prob(p::QBasedPolicy, env, ::MinimalActionSet) =
    get_prob(p.explorer, p.learner(env))
RLBase.get_prob(p::QBasedPolicy, env, ::FullActionSet) =
    get_prob(p.explorer, p.learner(env), get_legal_actions_mask(env))

@forward QBasedPolicy.learner RLBase.get_priority, RLBase.update!

function Flux.testmode!(p::QBasedPolicy, mode = true)
    testmode!(p.learner, mode)
    testmode!(p.explorer, mode)
end
