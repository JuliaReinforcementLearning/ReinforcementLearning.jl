using Random

#####
# Agent
#####

"""
An agent is a functional object which takes in an observation and returns an action.
"""
@interface abstract type AbstractAgent end
@interface (agent::AbstractAgent)(obs)

"""
Only do dry-run, shouldn't have any side-effect.
"""
@interface predict(agent::AbstractAgent, obs)

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

"""
Only do dry-run, shouldn't have any side-effect.
"""
@interface predict(p::AbstractExplorer, x)

#####
# Policy
#####

"""
A policy is a functional object which defines how to generate action(s) given an observation of the environment.
"""
@interface abstract type AbstractPolicy end

@interface (p::AbstractPolicy)(x)
@interface predict(p::AbstractPolicy, x)
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

@interface get_trace(t::AbstractTrajectory, s::Symbol)
@interface get_traces(t::AbstractTrajectory{names}) where {names} = Tuple(get_traces(t, s) for s in names)
@interface isfull(t::AbstractTrajectory) = all(isfull(x) for x in get_traces(t))

@interface Base.length(t::AbstractTrajectory) = maximum(length(x) for x in get_traces(t))
@interface Base.size(t::AbstractTrajectory) = (length(t),)
@interface Base.lastindex(t::AbstractTrajectory) = length(t)
@interface Base.isempty(t::AbstractTrajectory) = all(isempty(x) for x in get_traces(t))
@interface Base.getindex(t::AbstractTrajectory, s::Symbol) = get_trace(t, s)
@interface Base.getindex(b::AbstractTrajectory{names,types}, i::Int) where {names,types} = NamedTuple{names,types}(Tuple(x[i] for x in get_traces(b)))

@interface function Base.empty!(t::AbstractTrajectory)
    for x in get_traces(t)
        empty!(x)
    end
end

@interface function Base.push!(t::AbstractTrajectory{names}; kw...) where {names}
    for (k, v) in kw
        push!(t[k], v)
    end
end

@interface function Base.push!(t::AbstractTrajectory{names}, args...) where {names}
    for (k, v) in zip(names, args)
        push!(t[k], v)
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
"""
@interface abstract type AbstractEnvironmentModel end

@interface update!(m::AbstractEnvironmentModel, transition)

"""
For sample based models, we can sample a random transition from them.
"""
@interface abstract type AbstractSampleBasedModel <: AbstractEnvironmentModel end
@interface Base.rand(rng::AbstractRNG, m::AbstractSampleBasedModel)

"""
For distribution based models, we can get the reward distribution given a state and an action.
"""
@interface abstract type AbstractDistributionBasedModel <: AbstractEnvironmentModel end
@interface get_dist(m::AbstractDistributionBasedModel, s, a)

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

"""
Determine whether the players can play simultaneous or not.
"""
@interface abstract type AbstractDynamicStyle end
@interface struct Sequential <: AbstractDynamicStyle end
@interface const SEQUENTIAL = Sequential()
@interface struct Simultaneous <: AbstractDynamicStyle end
@interface const SIMULTANEOUS = Simultaneous()
@interface DynamicStyle(x)

@interface (env::AbstractEnv)(action)
@interface get_current_player(env::AbstractEnv)
@interface observe(env) = observe(env, get_current_player(env))
@interface observe(::AbstractEnv, players)
@interface action_space(::AbstractEnv)
@interface observation_space(::AbstractEnv)
@interface render(::AbstractEnv)

@interface reset!(::AbstractEnv)
@interface Random.seed!(::AbstractEnv, seed)

@interface max_utility(::AbstractEnv)
@interface min_utility(::AbstractEnv)

#####
# Observation
# !!!
# This is a very deliberate decision to adopt the duck-typing here to describe an observation from an environment.
# By default, we assume an observation is a NamedTuple, which is the most common case.
#####

"""
    ActionStyle(x)

Specify whether the observation contains a full action set or a minimal action set.

!!! note
    By default, we check if `x` has a property named `:legal_actions` or `legal_actions_mask`.
    So this is a dynamic dispatch. For performance, you may want to write a static method
    based on your customized type.
"""
@interface abstract type AbstractActionStyle end
@interface struct FullActionSet <: AbstractActionStyle end
@interface const FULL_ACTION_SET = FullActionSet()
@interface struct MinimalActionSet <: AbstractActionStyle end
@interface const MINIMAL_ACTION_SET = MinimalActionSet()

@interface function ActionStyle(x)
    if hasproperty(x, :legal_actions_mask) || hasproperty(x, :legal_actions)
        FullActionSet()
    else
        MinimalActionSet()
    end
end


@interface legal_actions(x) = x.legal_actions
@interface legal_actions_mask(x) = x.legal_actions_mask

@interface is_terminal(x) = x.is_terminal
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
@interface Base.getindex(s::AbstractSpace, i...)
@interface Base.in(x, s::AbstractSpace)
@interface Base.rand(rng::AbstractRNG, s::AbstractSpace)
@interface Base.eltype(s::AbstractSpace)

@interface element_size(s::AbstractSpace)
@interface element_length(s::AbstractSpace) = reduce(*, element_size(s))
