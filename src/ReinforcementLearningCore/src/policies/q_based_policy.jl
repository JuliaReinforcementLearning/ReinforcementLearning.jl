export QBasedPolicy

using Flux

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

Flux.@layer QBasedPolicy trainable=(learner,)

function RLBase.plan!(policy::QBasedPolicy{L,Ex}, env::E) where {Ex<:AbstractExplorer,L<:AbstractLearner,E<:AbstractEnv}
    RLBase.plan!(policy.explorer, policy.learner, env)
end

function RLBase.plan!(policy::QBasedPolicy{L,Ex}, env::E, player::Symbol) where {Ex<:AbstractExplorer,L<:AbstractLearner,E<:AbstractEnv}
    RLBase.plan!(policy.explorer, policy.learner, env, player)
end

RLBase.prob(policy::QBasedPolicy{L,Ex}, env::AbstractEnv) where {L<:AbstractLearner,Ex<:AbstractExplorer} =
    prob(policy.explorer, forward(policy.learner, env), legal_action_space_mask(env))

#the internal learner defines the optimization stage.
RLBase.optimise!(policy::QBasedPolicy, stage::AbstractStage, trajectory::Trajectory) = RLBase.optimise!(policy.learner, stage, trajectory)
