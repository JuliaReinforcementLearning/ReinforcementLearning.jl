export AbstractLearner, Approximator

using Flux

"""
    AbstractLearner

Abstract type for a learner.
"""
abstract type AbstractLearner end

Base.show(io::IO, m::MIME"text/plain", learner::AbstractLearner) = show(io, m, convert(AnnotatedStructTree, learner))

# Take Learner and Environment, get state, send to RLCore.forward(Learner, State)
function forward(learner::L, env::E) where {L <: AbstractLearner, E <: AbstractEnv}
    env |> state |> (x -> forward(learner, x))
end

# Take Learner and Environment, get state, send to RLCore.forward(Learner, State)
function forward(learner::L, env::E, player::Player) where {L <: AbstractLearner, E <: AbstractEnv, Player <: AbstractPlayer}
    env |> (x -> state(x, player)) |> (x -> forward(learner, x))
end

function RLBase.optimise!(::AbstractLearner, ::AbstractStage, ::Trajectory) end

function RLBase.optimise!(::AbstractLearner, ::AbstractStage, ::NamedTuple) end

function RLBase.plan!(explorer::AbstractExplorer, learner::AbstractLearner, env::AbstractEnv, player=current_player(env))
    return RLBase.plan!(ActionStyle(env), explorer, learner, env, player)
end

function RLBase.plan!(::FullActionSet, explorer::AbstractExplorer, learner::AbstractLearner, env::AbstractEnv, player=current_player(env))
    legal_action_space_ = RLBase.legal_action_space_mask(env, player)
    return RLBase.plan!(explorer, forward(learner, env, player), legal_action_space_)
end

function RLBase.plan!(::MinimalActionSet, explorer::AbstractExplorer, learner::AbstractLearner, env::AbstractEnv, player=current_player(env))
    return RLBase.plan!(explorer, forward(learner, env, player))
end
