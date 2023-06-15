export QBasedPolicy

include("learners.jl")
include("explorers/explorers.jl")

using Functors: @functor

"""
    QBasedPolicy(;learner, explorer)
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

    RLBase.optimise!(p::QBasedPolicy{L,Ex}, stage::S, t::Trajectory) where {L<:AbstractLearner,Ex<:AbstractExplorer, S<:AbstractStage} = optimise!(p.learner, stage, t)
