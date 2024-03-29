export AbstractLearner, Approximator

using Flux

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

function RLBase.plan!(explorer::AbstractExplorer, learner::AbstractLearner, env::AbstractEnv)
    legal_action_space_ = RLBase.legal_action_space_mask(env)
    RLBase.plan!(explorer, forward(learner, env), legal_action_space_)
end

function RLBase.plan!(explorer::AbstractExplorer, learner::AbstractLearner, env::AbstractEnv, player::AbstractPlayer)
    legal_action_space_ = RLBase.legal_action_space_mask(env, player)
    return RLBase.plan!(explorer, forward(learner, env, player), legal_action_space_)
end
