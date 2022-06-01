export QBasedPolicy

include("learners.jl")
include("explorers/explorers.jl")

import Functors

Base.@kwdef mutable struct QBasedPolicy{L,E} <: AbstractPolicy
    learner::L
    explorer::E
end

Functors.functor(x::QBasedPolicy) =
    (learner = x.learner,), y -> QBasedPolicy(y.learner, x.explorer)

(p::QBasedPolicy)(env) = p.explorer(p.learner(env), legal_action_space_mask(env))

RLBase.prob(p::QBasedPolicy, env::AbstractEnv) =
    prob(p.explorer, p.learner(env), legal_action_space_mask(env))

RLBase.optimise!(p::QBasedPolicy, x::NamedTuple) = optimise!(p.learner, x)
