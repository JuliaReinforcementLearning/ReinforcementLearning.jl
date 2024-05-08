export QBasedPolicy

using Flux

"""
    QBasedPolicy(;learner, explorer)

Wraps a learner and an explorer. The learner is a struct that should predict the Q-value of each legal
action of an environment at its current state. It is typically a table or a neural network. 
QBasedPolicy can be queried for an action with `RLBase.plan!`, the explorer will affect the action selection
accordingly.
"""
struct QBasedPolicy{L<:AbstractLearner,E<:AbstractExplorer} <: AbstractPolicy
    "estimate the Q value"
    learner::L
    "select the action based on Q values calculated by the learner"
    explorer::E

    function QBasedPolicy(; learner::L, explorer::E) where {L<:AbstractLearner, E<:AbstractExplorer}
        new{L,E}(learner, explorer)
    end

    function QBasedPolicy(learner::L, explorer::E) where {L<:AbstractLearner, E<:AbstractExplorer}
        new{L,E}(learner, explorer)
    end
end

Flux.@layer QBasedPolicy trainable=(learner,)

function RLBase.plan!(policy::QBasedPolicy{L,Ex}, env::E) where {Ex<:AbstractExplorer,L<:AbstractLearner,E<:AbstractEnv}
    RLBase.plan!(policy.explorer, policy.learner, env)
end

function RLBase.plan!(policy::QBasedPolicy{L,Ex}, env::E, player::Player) where {Ex<:AbstractExplorer,L<:AbstractLearner,E<:AbstractEnv, Player<:AbstractPlayer}
    RLBase.plan!(policy.explorer, policy.learner, env, player)
end

RLBase.prob(policy::QBasedPolicy{L,Ex}, env::AbstractEnv) where {L<:AbstractLearner,Ex<:AbstractExplorer} =
    prob(ActionStyle(env), policy, env)

RLBase.prob(::MinimalActionSet, policy::QBasedPolicy{L,Ex}, env::AbstractEnv) where {L<:AbstractLearner,Ex<:AbstractExplorer} =
    prob(policy.explorer, forward(policy.learner, env))

RLBase.prob(::FullActionSet, policy::QBasedPolicy{L,Ex}, env::AbstractEnv) where {L<:AbstractLearner,Ex<:AbstractExplorer} =
    prob(policy.explorer, forward(policy.learner, env), legal_action_space_mask(env))


#the internal learner defines the optimization stage.
RLBase.optimise!(policy::QBasedPolicy, stage::AbstractStage, trajectory::Trajectory) = RLBase.optimise!(policy.learner, stage, trajectory)
