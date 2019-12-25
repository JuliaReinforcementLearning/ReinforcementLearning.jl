using Random

#####
# Agent
#####

"An agent is a functional object which takes in an observation and returns an action."
@interface abstract type AbstractAgent end
@interface (agent::AbstractAgent)(obs)

#####
# Approximator
#####

"""
An approximator is a functional object for value estimation.
"""
@interface abstract type AbstractApproximator end
@interface (app::AbstractApproximator)(x)
@interface update!(a::AbstractApproximator, correction)

#####
# Learner
#####

"""
A learner is usually a wrapper around [`AbstractApproximator`](@ref)s.
It defines the expected inputs and how to udpate inner approximators.
"""
@interface abstract type AbstractLearner end
@interface (learner::AbstractLearner)(x)
@interface update!(learner::AbstractLearner, experience)

#####
# Explorer
#####

"""
Define how to select actions.
"""
@interface abstract type AbstractExplorer end
@interface (p::AbstractExplorer)(x)
@interface reset!(p::AbstractExplorer)
@interface Base.copy(p::AbstractExplorer)

"Get the action distribution given action values"
@interface get_distribution(p::AbstractExplorer, x)

#####
# Policy
#####

"""
A policy is a functional object which defines how to generate action(s)
given an observation of the environment.
"""
@interface abstract type AbstractPolicy end
@interface (p::AbstractPolicy)(x)
@interface update!(p::AbstractPolicy, experience)

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
@interface abstract type AbstractTrajectory{names,types} <: AbstractArray{NamedTuple{names,types},1} end

# some typical trace names
@interface const RTSA = (:reward, :terminal, :state, :action)
@interface const SARTSA = (:state, :action, :reward, :terminal, :next_state, :next_action)

@interface get_trace(t::AbstractTrajectory, s::Symbol)
@interface get_traces(t::AbstractTrajectory{names}) where {names} = merge(NamedTuple(), (s, get_trace(t, s)) for s in names)

@interface Base.length(t::AbstractTrajectory) = maximum(length(x) for x in get_traces(t))
@interface Base.size(t::AbstractTrajectory) = (length(t),)
@interface Base.lastindex(t::AbstractTrajectory) = length(t)
@interface Base.getindex(t::AbstractTrajectory{names,types}, i::Int) where {names,types} = NamedTuple{names,types}(Tuple(x[i] for x in get_traces(t)))

@interface Base.isempty(t::AbstractTrajectory) = all(isempty(t) for t in get_traces(t))
@interface isfull(t::AbstractTrajectory) = all(isfull(x) for x in get_traces(t))

@interface function Base.empty!(t::AbstractTrajectory)
    for x in get_traces(t)
        empty!(x)
    end
end

@interface function Base.push!(t::AbstractTrajectory;kwargs...)
    for (k, v) in kwargs
        push!(get_trace(t, k), v)
    end
end

"""
    extract_transitions(trajectory::AbstractTrajectory, learner::AbstractLearner)

Extract transitions given a `learner`. Then the result is used to update the `learner`.
"""
@interface extract_transitions(trajectory::AbstractTrajectory, learner::AbstractLearner)

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
@interface (p::AbstractPreprocessor)(x)

#####
# Environment
#####

"""
Super type of all reinforcement learning environments.
"""
@interface abstract type AbstractEnv end
@interface (env::AbstractEnv)(action)

"""
Determine whether the players can play simultaneous or not.
"""
abstract type AbstractDynamicStyle end

@interface struct Sequential <: AbstractDynamicStyle end
@interface const SEQUENTIAL = Sequential()
@interface struct Simultaneous <: AbstractDynamicStyle end
@interface const SIMULTANEOUS = Simultaneous()
@interface DynamicStyle(x::AbstractEnv) = SEQUENTIAL

struct DefaultPlayer end
@interface const DEFAULT_PLAYER = DefaultPlayer()
@interface get_current_player(env::AbstractEnv) = DEFAULT_PLAYER
@interface observe(env::AbstractEnv) = observe(env, get_current_player(env))
@interface observe(::AbstractEnv, players)
@interface get_action_space(env::AbstractEnv) = env.action_space
@interface get_observation_space(env::AbstractEnv) = env.observation_space
@interface render(::AbstractEnv)

@interface reset!(::AbstractEnv)
@interface Random.seed!(::AbstractEnv, seed)

#####
# Observation
# !!!
# This is a very deliberate decision to adopt the duck-typing here to describe an observation from an environment.
# By default, we assume an observation is a NamedTuple, which is the most common case.
#####

abstract type AbstractActionSet end
@interface struct FullActionSet <: AbstractActionSet end
@interface const FULL_ACTION_SET = FullActionSet()
@interface struct MinimalActionSet <: AbstractActionSet end
@interface const MINIMAL_ACTION_SET = MinimalActionSet()

"""
    ActionStyle(x)

Specify whether the observation contains a full action set or a minimal action set.
"""
@interface ActionStyle(::NamedTuple{(:reward, :terminal, :state)}) = MINIMAL_ACTION_SET
@interface ActionStyle(::NamedTuple{(:reward, :terminal, :state, :legal_actions)}) = FULL_ACTION_SET
@interface ActionStyle(::NamedTuple{(:reward, :terminal, :state, :legal_actions, :legal_actions_mask)}) = FULL_ACTION_SET


@interface legal_actions(x) = findall(legal_actions_mask(x))
@interface legal_actions_mask(x) = x.legal_actions_mask

@interface is_terminal(x) = x.terminal
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

@interface element_size(s::AbstractSpace)
@interface element_length(s::AbstractSpace) = reduce(*, element_size(s))
