using Random
using Distributions: pdf

#####
# general
#####

@interface abstract type AbstractStage end

@interface struct PreExperimentStage <: AbstractStage end
@interface struct PostExperimentStage <: AbstractStage end
@interface struct PreEpisodeStage <: AbstractStage end
@interface struct PostEpisodeStage <: AbstractStage end
@interface struct PreActStage <: AbstractStage end
@interface struct PostActStage <: AbstractStage end

@interface const PRE_EXPERIMENT_STAGE = PreExperimentStage()
@interface const POST_EXPERIMENT_STAGE = PostExperimentStage()
@interface const PRE_EPISODE_STAGE = PreEpisodeStage()
@interface const POST_EPISODE_STAGE = PostEpisodeStage()
@interface const PRE_ACT_STAGE = PreActStage()
@interface const POST_ACT_STAGE = PostActStage()

struct DefaultPlayer end
@interface const DEFAULT_PLAYER = DefaultPlayer()

abstract type AbstractActionStyle end
@interface struct FullActionSet <: AbstractActionStyle end
@interface const FULL_ACTION_SET = FullActionSet()
@interface struct MinimalActionSet <: AbstractActionStyle end
@interface const MINIMAL_ACTION_SET = MinimalActionSet()

"""
    ActionStyle(x)

Specify whether the observation contains a full action set or a minimal action set.
By default the `MINIMAL_ACTION_SET` is returned.
"""
@interface ActionStyle(x) = MINIMAL_ACTION_SET

ActionStyle(::NamedTuple{(:reward, :terminal, :state, :legal_actions)}) =
    FULL_ACTION_SET
ActionStyle(::NamedTuple{(:reward, :terminal, :state, :legal_actions_mask)}) =
    FULL_ACTION_SET
ActionStyle(
    ::NamedTuple{(:reward, :terminal, :state, :legal_actions, :legal_actions_mask)},
) = FULL_ACTION_SET


#####
# Agent
#####

"An agent is a functional object which takes in an observation and returns an action."
@interface abstract type AbstractAgent end

@interface (agent::AbstractAgent)(obs) = agent(PRE_ACT_STAGE, obs)
@interface (agent::AbstractAgent)(stage::AbstractStage, obs)
@interface get_role(::AbstractAgent) = DEFAULT_PLAYER

#####
# Approximator
#####

abstract type AbstractApproximatorStyle end

"""
For `VApproximator`, we assume that `(V::AbstractApproximator)(s)` is implemented.
"""
@interface struct VApproximator <: AbstractApproximatorStyle end

"""
For `QApproximator`, we assume that the following methods are implemented:

- `(Q::AbstractApproximator)(s, a)`, estimate the Q value.
- `(Q::AbstractApproximator)(s)`, estimate the Q value among all possible actions.
"""
@interface struct QApproximator <: AbstractApproximatorStyle end

"""
For `HybridApproximator`, the following methods are assumed to be implemented:
- `(Q::AbstractApproximator)(s, a)`, estimate the Q value.
- `(Q::AbstractApproximator)(s)`, estimate the state value.
"""
@interface struct HybridApproximator <: AbstractApproximatorStyle end

"""
An approximator is a functional object for value estimation.
"""
@interface abstract type AbstractApproximator end
@interface (app::AbstractApproximator)(obs) = app(get_state(obs))

"Usually the `correction` is the gradient of inner parameters"
@interface update!(a::AbstractApproximator, correction)

@interface ApproximatorStyle(x::AbstractApproximator)

#####
# Learner
#####

"""
A learner is usually a wrapper around [`AbstractApproximator`](@ref)s.
It defines the expected inputs and how to udpate inner approximators.
"""
@interface abstract type AbstractLearner end
@interface (learner::AbstractLearner)(obs)
@interface update!(learner::AbstractLearner, experience)
@interface get_priority(p::AbstractLearner, experience)

#####
# Explorer
#####

"""
Define how to select an action based on action values.
"""
@interface abstract type AbstractExplorer end
@interface (p::AbstractExplorer)(x)
@interface (p::AbstractExplorer)(x, mask)
@interface reset!(p::AbstractExplorer)
@interface Base.copy(p::AbstractExplorer)

"Get the action distribution given action values"
@interface get_prob(p::AbstractExplorer, x)
@interface get_prob(p::AbstractExplorer, x, mask)

#####
# Policy
#####

"""
A policy is a functional object which defines how to generate action(s)
given an observation of the environment.
"""
@interface abstract type AbstractPolicy end
@interface (p::AbstractPolicy)(obs) = p(obs, ActionStyle(obs))
@interface update!(p::AbstractPolicy, experience)

@interface get_prob(p::AbstractPolicy, obs) = get_prob(p, obs, ActionStyle(obs))
@interface get_prob(p::AbstractPolicy, obs, ::AbstractActionStyle)
@interface get_prob(p::AbstractPolicy, obs, a) = get_prob(p, obs, ActionStyle(obs), a)
@interface get_prob(p::AbstractPolicy, obs, ::AbstractActionStyle, a) = pdf(get_prob(p, obs), a)

@interface get_priority(p::AbstractPolicy, experience)

#####
# Trajectory
#####

"""
Record useful information during the interactions between agents and environments.

# Parameters

- `names`::`NTuple{Symbol}`, indicate what fields to be recorded.
- `types`::`Tuple{DataType...}`, the datatypes of `names`.

The length of `names` and `types` must match.
"""
@interface abstract type AbstractTrajectory{names,types} <:
                         AbstractArray{NamedTuple{names,types},1} end

# some typical trace names
@interface const RTSA = (:reward, :terminal, :state, :action)
@interface const SARTSA = (:state, :action, :reward, :terminal, :next_state, :next_action)

@interface get_trace(t::AbstractTrajectory, s::Symbol)
@interface get_trace(t::AbstractTrajectory, s::Symbol...) = merge(NamedTuple(), (x, get_trace(t, x)) for x in s)
@interface get_trace(t::AbstractTrajectory{names}) where {names} =
    merge(NamedTuple(), (s, get_trace(t, s)) for s in names)

@interface Base.length(t::AbstractTrajectory) = maximum(length(x) for x in get_trace(t))
@interface Base.size(t::AbstractTrajectory) = (length(t),)
@interface Base.lastindex(t::AbstractTrajectory) = length(t)
@interface Base.getindex(t::AbstractTrajectory{names,types}, i::Int) where {names,types} =
    NamedTuple{names,types}(Tuple(x[i] for x in get_trace(t)))

@interface Base.isempty(t::AbstractTrajectory) = all(isempty(t) for t in get_trace(t))

@interface function Base.empty!(t::AbstractTrajectory)
    for x in get_trace(t)
        empty!(x)
    end
end

@interface function Base.push!(t::AbstractTrajectory; kwargs...)
    for kv in kwargs
        push!(t, kv)
    end
end

@interface Base.push!(t::AbstractTrajectory, kv::Pair{Symbol})

@interface function Base.pop!(t::AbstractTrajectory{names}) where {names}
    pop!(t, names...)
end

@interface Base.pop!(t::AbstractTrajectory, s::Symbol...)

@interface extract_experience(trajectory::AbstractTrajectory, learner::AbstractLearner)

#####
# EnvironmentModel
#####

"""
Describe how to model a reinforcement learning environment.

TODO: need more investigation

Ref: https://bair.berkeley.edu/blog/2019/12/12/mbpo/
- Analytic gradient computation
- Sampling-based planning
- Model-based data generation
- Value-equivalence prediction
"""
@interface abstract type AbstractEnvironmentModel end

#####
# Preprocessor
#####

"""
Preprocess an observation and return a new observation.
"""
@interface abstract type AbstractPreprocessor end

#####
# Environment
#####

"""
Super type of all reinforcement learning environments.
"""
@interface abstract type AbstractEnv end
"""
Determine whether the players can play simultaneous or not.
"""
abstract type AbstractDynamicStyle end

@interface struct Sequential <: AbstractDynamicStyle end
@interface const SEQUENTIAL = Sequential()
@interface struct Simultaneous <: AbstractDynamicStyle end
@interface const SIMULTANEOUS = Simultaneous()
@interface DynamicStyle(x::AbstractEnv) = SEQUENTIAL

@interface (env::AbstractEnv)(action) = env(DEFAULT_PLAYER, action)
@interface (env::AbstractEnv)(player, action)

@interface observe(env::AbstractEnv) = observe(env, get_current_player(env))
@interface observe(::AbstractEnv, player)
@interface get_action_space(env::AbstractEnv) = env.action_space
@interface get_observation_space(env::AbstractEnv) = env.observation_space
@interface get_current_player(env::AbstractEnv) = DEFAULT_PLAYER
@interface get_num_players(env::AbstractEnv) = 1
@interface render(::AbstractEnv)

@interface reset!(::AbstractEnv)
@interface Random.seed!(::AbstractEnv, seed)

#####
# Observation
# !!!
# This is a very deliberate decision to adopt the duck-typing here to describe an observation from an environment.
# By default, we assume an observation is a NamedTuple, which is the most common case.
#####


@interface get_legal_actions_mask(x) = x.legal_actions_mask
@interface get_legal_actions(x) = findall(get_legal_actions_mask(x))

get_legal_actions(x::NamedTuple{(:reward, :terminal, :state, :legal_actions, :legal_actions_mask)}) = x.legal_actions
get_legal_actions(x::NamedTuple{(:reward, :terminal, :state, :legal_actions)}) = x.legal_actions


@interface get_terminal(x) = x.terminal
@interface get_reward(x) = x.reward
@interface get_state(x) = x.state

#####
# Space
#####

"""
Describe the span of observations and actions.
"""
@interface abstract type AbstractSpace end

@interface Base.length(::AbstractSpace)
@interface Base.in(x, s::AbstractSpace)
@interface Base.rand(rng::AbstractRNG, s::AbstractSpace)
@interface Base.eltype(s::AbstractSpace)