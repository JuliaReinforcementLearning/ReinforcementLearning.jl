using ReinforcementLearningCore

using AbstractTrees

"""
    EnrichedAction(action;kwargs...)

Inject some runtime info into the action
"""
struct EnrichedAction{A,M}
    action::A
    meta::M
end

EnrichedAction(action; kwargs...) = EnrichedAction(action, values(kwargs))

(env::AbstractEnv)(action::EnrichedAction) = env(action.action)
