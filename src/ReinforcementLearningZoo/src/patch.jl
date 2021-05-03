using ReinforcementLearningCore

using AbstractTrees
using TensorBoardLogger: TBLogger

RLCore.is_expand(::TBLogger) = false

"""
    EnrichedAction(action;kwargs...)

Inject some runtime info into the action
"""
struct EnrichedAction{A,M}
    action::A
    meta::M
end

EnrichedAction(action; kwargs...) = EnrichedAction(action, kwargs.data)

(env::AbstractEnv)(action::EnrichedAction) = env(action.action)
