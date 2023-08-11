export AbstractLearner, Approximator

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

Base.@kwdef mutable struct Approximator{M,O}
    model::M
    optimiser::O
end

Base.show(io::IO, m::MIME"text/plain", A::Approximator) = show(io, m, convert(AnnotatedStructTree, A))

@functor Approximator (model,)

forward(A::Approximator, args...; kwargs...) = A.model(args...; kwargs...)

RLBase.optimise!(A::Approximator, gs) = Flux.Optimise.update!(A.optimiser, Flux.params(A), gs)
