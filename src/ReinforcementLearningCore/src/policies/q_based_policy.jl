export QBasedPolicy

using Functors: @functor

"""
    QBasedPolicy(;learner, explorer)

Wraps a learner and an explorer. The learner is a struct that should predict the Q-value of each legal
action of an environment at its current state. It is typically a table or a neural network. 
QBasedPolicy can be queried for an action with `RLBase.plan!`, the explorer will affect the action selection
accordingly.
"""
Base.@kwdef mutable struct QBasedPolicy{L,E} <: AbstractPolicy
    "estimate the Q value"
    learner::L
    "select the action based on Q values calculated by the learner"
    explorer::E
end

@functor QBasedPolicy (learner,)

function RLBase.plan!(p::QBasedPolicy{L,Ex}, env::E) where {Ex<:AbstractExplorer,L<:AbstractLearner,E<:AbstractEnv}
    RLBase.plan!(p.explorer, p.learner, env)
end

function RLBase.plan!(explorer::Ex, learner::L, env::E) where {Ex<:AbstractExplorer,L<:AbstractLearner,E<:AbstractEnv}
    RLBase.plan!(explorer, forward(learner, env), legal_action_space_mask(env))
end

function RLBase.plan!(p::QBasedPolicy{L,Ex}, env::E, player::Symbol) where {Ex<:AbstractExplorer,L<:AbstractLearner,E<:AbstractEnv}
    RLBase.plan!(p.explorer, p.learner, env, player)
end

function RLBase.plan!(explorer::Ex, learner::L, env::E, player::Symbol) where {Ex<:AbstractExplorer,L<:AbstractLearner,E<:AbstractEnv}
    legal_action_space_ = RLBase.legal_action_space_mask(env, player)
    return RLBase.plan!(explorer, forward(learner, env), legal_action_space_)
end

RLBase.prob(p::QBasedPolicy{L,Ex}, env::AbstractEnv) where {L<:AbstractLearner,Ex<:AbstractExplorer} =
    prob(p.explorer, forward(p.learner, env), legal_action_space_mask(env))

#the internal learner defines the optimization stage.
RLBase.optimise!(p::QBasedPolicy, s::AbstractStage, trajectory::Trajectory) = RLBase.optimise!(p.learner, s, trajectory)
