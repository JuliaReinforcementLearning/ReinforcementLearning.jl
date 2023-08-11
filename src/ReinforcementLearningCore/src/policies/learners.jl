export AbstractLearner

import Flux
using Functors: @functor

abstract type AbstractLearner end

function RLBase.plan!(explorer::AbstractExplorer, learner::AbstractLearner, env::AbstractEnv)
    legal_action_space_ = RLBase.legal_action_space_mask(env)
    RLBase.plan!(explorer, forward(learner, env), legal_action_space_)
end

function RLBase.plan!(explorer::AbstractExplorer, learner::AbstractLearner, env::AbstractEnv, player::Symbol)
    legal_action_space_ = RLBase.legal_action_space_mask(env, player)
    return RLBase.plan!(explorer, forward(learner, env), legal_action_space_)
end

# Take Learner and Environment, get state, send to RLCore.forward(Learner, State)
function forward(L::AbstractLearner, env::AbstractEnv)
    s = state(env) |> send_to_device(L.approximator) 
    forward(L,s) |> send_to_device(env) 
end

function RLBase.optimise!(::AbstractLearner, ::AbstractStage, ::Trajectory) end

Base.show(io::IO, m::MIME"text/plain", L::AbstractLearner) = show(io, m, convert(AnnotatedStructTree, L))
