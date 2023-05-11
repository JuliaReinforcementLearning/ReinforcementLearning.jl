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

RLBase.plan!(p::QBasedPolicy, env::E) where {E <: AbstractEnv} = p.explorer(p.learner(env), legal_action_space_mask(env))

function RLBase.plan!(p::QBasedPolicy, env::E, player::Symbol) where {E<:AbstractEnv}
    legal_action_space_ = RLBase.legal_action_space_mask(env, player)
    return p.explorer(p.learner(env), legal_action_space_)
end

RLBase.prob(p::QBasedPolicy, env::AbstractEnv) =
    prob(p.explorer, p.learner(env), legal_action_space_mask(env))

RLBase.optimise!(p::QBasedPolicy, x::NamedTuple) = optimise!(p.learner, x)
