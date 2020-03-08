@doc """
[ReinforcementLearningBase.jl](https://github.com/JuliaReinforcementLearning/ReinforcementLearningBase.jl) (**RLBase**)
provides some common constants, traits, abstractions and interfaces in developing reinforcement learning algorithms in 
Julia. From the concept level, they can be organized in the following parts:

- [General](@ref)
- [Agent](@ref)
  - [Policy](@ref)
    - [Learner](@ref)
      - [Approximator](@ref)
    - [Explorer](@ref)
- [EnvironmentModel](@ref)
- [Environment](@ref)
  - [Preprocessor](@ref)
  - [Observation](@ref)
  - [Trajectory](@ref)
  - [Space](@ref)

""" RLBase

using Random

#####
# general
#####

"""
In the simplest case, we just need to run `env |> observe |> agent |> env` repeatly. But usually we would also like to control when/how to update the agent. So we defined the following stages for the help:

- `PRE_EXPERIMENT_STAGE`
- `PRE_EPISODE_STAGE`
- `PRE_ACT_STAGE`
- `POST_ACT_STAGE`
- `POST_EPISODE_STAGE`
- `POST_EXPERIMENT_STAGE`

```
                      +-----------------------------------------------------------+                      
                      |Episode                                                    |                      
                      |                                                           |                      
PRE_EXPERIMENT_STAGE  |            PRE_ACT_STAGE    POST_ACT_STAGE                | POST_EXPERIMENT_STAGE
         |            |                  |                |                       |          |           
         v            |        +-----+   v   +-------+    v   +-----+             |          v           
         --------------------->+ env +------>+ agent +------->+ env +---> ... ------->......             
                      |  ^     +-----+  obs  +-------+ action +-----+          ^  |                      
                      |  |                                                     |  |                      
                      |  +--PRE_EPISODE_STAGE            POST_EPISODE_STAGE----+  |                      
                      |                                                           |                      
                      |                                                           |                      
                      +-----------------------------------------------------------+                      
```
"""
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

"""
The default value of invalid action for all the environments.
However, one can still specify the value for a specific environment
or observation.

# See also: [`get_invalid_action`](@ref)
"""
@interface const INVALID_ACTION = 0

struct DefaultPlayer end

"Default player instance for all the environments"
@interface const DEFAULT_PLAYER = DefaultPlayer()

abstract type AbstractActionStyle end
@interface struct FullActionSet <: AbstractActionStyle end

"The action space of the environment may contains illegal actions"
@interface const FULL_ACTION_SET = FullActionSet()

@interface struct MinimalActionSet <: AbstractActionStyle end

"All actions in the action space of the environment are legal"
@interface const MINIMAL_ACTION_SET = MinimalActionSet()

"""
    ActionStyle(x)

Specify whether the observation contains a full action set or a minimal action set.
By default the [`MINIMAL_ACTION_SET`](@ref) is returned.
"""
@interface ActionStyle(x) = MINIMAL_ACTION_SET

ActionStyle(::NamedTuple{(:reward, :terminal, :state, :legal_actions)}) = FULL_ACTION_SET
ActionStyle(::NamedTuple{(:reward, :terminal, :state, :legal_actions_mask)}) =
    FULL_ACTION_SET
ActionStyle(
    ::NamedTuple{(:reward, :terminal, :state, :legal_actions, :legal_actions_mask)},
) = FULL_ACTION_SET


#####
# Agent
#####

"""
    (agent::AbstractAgent)(obs) = agent(PRE_ACT_STAGE, obs) -> action
    (agent::AbstractAgent)(stage::AbstractStage, obs)

An agent is a functional object which takes in an observation and returns an action.
In different stages, the behavior may be different.
"""
@interface abstract type AbstractAgent end

@interface (agent::AbstractAgent)(obs) = agent(PRE_ACT_STAGE, obs)
@interface (agent::AbstractAgent)(stage::AbstractStage, obs)

"return [`DEFAULT_PLAYER`](@ref) by default"
@interface get_role(::AbstractAgent) = DEFAULT_PLAYER

#####
# Approximator
#####

abstract type AbstractApproximatorStyle end

@interface struct VApproximator <: AbstractApproximatorStyle end

"""
For `V_APPROXIMATOR`, we assume that `(V::AbstractApproximator)(s)` is implemented.
"""
@interface const V_APPROXIMATOR = VApproximator()

@interface struct QApproximator <: AbstractApproximatorStyle end

"""
For `Q_APPROXIMATOR`, we assume that the following methods are implemented:

- `(Q::AbstractApproximator)(s, a)`, estimate the Q value.
- `(Q::AbstractApproximator)(s)`, estimate the Q value among all possible actions.
"""
@interface const Q_APPROXIMATOR = QApproximator()

@interface struct HybridApproximator <: AbstractApproximatorStyle end

"""
For `HYBRID_APPROXIMATOR`, the following methods are assumed to be implemented:
- `(app::AbstractApproximator)(s, ::Val{:Q})`, estimate the action values of state `s`.
- `(app::AbstractApproximator)(s, ::Val{:V})`, estimate the state values.
- `(app::AbstractApproximator)(s, a)`, estimate state-action values.
"""
@interface const HYBRID_APPROXIMATOR = HybridApproximator()

"""
    (app::AbstractApproximator)(obs) = app(get_state(obs))

An approximator is a functional object for value estimation.
It serves as a black box to provides an abstraction over different 
kinds of approximate methods (for example DNN provided by Flux or Knet).
"""
@interface abstract type AbstractApproximator end
@interface (app::AbstractApproximator)(obs) = app(get_state(obs))

"""
    batch_estimate(app::AbstractApproximator, states)

The `states` is assume to be a batch of states
"""
@interface batch_estimate(app::AbstractApproximator, states)

"""
    Base.copyto!(dest::AbstractApproximator, src::AbstractApproximator)

Copy the internal params from `src` approximator to `dest` approximator
"""
@interface Base.copyto!(dest::AbstractApproximator, src::AbstractApproximator)

"""
    update!(a::AbstractApproximator, correction)

Usually the `correction` is the gradient of inner parameters.
"""
@interface update!(a::AbstractApproximator, correction)

"""
    ApproximatorStyle(x::AbstractApproximator)

Detect which kind of approximator it is
"""
@interface ApproximatorStyle(x::AbstractApproximator)

#####
# Learner
#####

"""
    (learner::AbstractLearner)(obs)

A learner is usually a wrapper around [`AbstractApproximator`](@ref)s.
It defines the expected inputs and how to update inner approximators.
From the concept level, it assumes that the necessary training data is 
already cooked (usually by [`AbstractPolicy`](@ref) with the [`extract_experience`](@ref) method).
"""
@interface abstract type AbstractLearner end
@interface (learner::AbstractLearner)(obs)

"""
    update!(learner::AbstractLearner, experience)
"""
@interface update!(learner::AbstractLearner, experience)

"""
    get_priority(p::AbstractLearner, experience)
"""
@interface get_priority(p::AbstractLearner, experience)

#####
# Explorer
#####

"""
    (p::AbstractExplorer)(x)
    (p::AbstractExplorer)(x, mask)

Define how to select an action based on action values.
"""
@interface abstract type AbstractExplorer end
@interface (p::AbstractExplorer)(x)
@interface (p::AbstractExplorer)(x, mask)

"Reset the internal state of an explorer `p`"
@interface reset!(p::AbstractExplorer)

"The internal state is also copied"
@interface Base.copy(p::AbstractExplorer)

"""
    get_prob(p::AbstractExplorer, x) -> AbstractDistribution

Get the action distribution given action values
"""
@interface get_prob(p::AbstractExplorer, x)

"""
    get_prob(p::AbstractExplorer, x, mask)

Similart to `get_prob(p::AbstractExplorer, x)`, but here only the `mask`ed elements are considered.
"""
@interface get_prob(p::AbstractExplorer, x, mask)

#####
# Policy
#####

"""
    (p::AbstractPolicy)(obs) = p(obs, ActionStyle(obs)) -> action

Similar to [`AbstractAgent`](@ref), a policy is aslo a functional object
which defines how to generate action(s) given an observation of the environment.
However, in the concept level, a policy is more lower and usually it doesn't need
to care about how/when to interact with an environment.
"""
@interface abstract type AbstractPolicy end
@interface (p::AbstractPolicy)(obs) = p(obs, ActionStyle(obs))

"""
    update!(p::AbstractPolicy, experience)

Update the policy `p` with experience
"""
@interface update!(p::AbstractPolicy, experience)

"""
    get_prob(p::AbstractPolicy, obs) = get_prob(p, obs, ActionStyle(obs))
    get_prob(p::AbstractPolicy, obs, ::AbstractActionStyle)
Get the probability distribution of actions from the policy `p` given an observation `obs`
"""
@interface get_prob(p::AbstractPolicy, obs) = get_prob(p, obs, ActionStyle(obs))
@interface get_prob(p::AbstractPolicy, obs, ::AbstractActionStyle)

"""
    get_prob(p::AbstractPolicy, obs, a) = get_prob(p, obs, ActionStyle(obs), a)

Get the probability of to take action `a` from the policy `p` given an observation `obs`
"""
@interface get_prob(p::AbstractPolicy, obs, a) = get_prob(p, obs, ActionStyle(obs), a)

"""
    get_priority(p::AbstractPolicy, experience)

Calculate the priority of the `experience` to policy `p`
"""
@interface get_priority(p::AbstractPolicy, experience)

#####
# Trajectory
#####

"""
    AbstractTrajectory{names,types} <: AbstractArray{NamedTuple{names,types},1}

A trajectory is used to record some useful information
during the interactions between agents and environments.

# Parameters

- `names`::`NTuple{Symbol}`, indicate what fields to be recorded.
- `types`::`Tuple{DataType...}`, the datatypes of `names`.

The length of `names` and `types` must match.

Required Methods:

- [`get_trace`](@ref)
- `Base.push!(t::AbstractTrajectory, kv::Pair{Symbol})`
- `Base.pop!(t::AbstractTrajectory, s::Symbol)`

Optional Methods:

- `Base.length`
- `Base.size`
- `Base.lastindex`
- `Base.isempty`
- `Base.empty!`
- `Base.push!`
- `Base.pop!`
"""
@interface abstract type AbstractTrajectory{names,types} <:
                         AbstractArray{NamedTuple{names,types},1} end

# some typical trace names
"An alias of `(:reward, :terminal, :state, :action)`"
@interface const RTSA = (:reward, :terminal, :state, :action)

"An alias of `(:state, :action, :reward, :terminal, :next_state, :next_action)`"
@interface const SARTSA = (:state, :action, :reward, :terminal, :next_state, :next_action)

"""
    get_trace(t::AbstractTrajectory, s::Symbol)
"""
@interface get_trace(t::AbstractTrajectory, s::Symbol)

"""
    get_trace(t::AbstractTrajectory, s::NTuple{N,Symbol}) where {N}
"""
get_trace(t::AbstractTrajectory, s::NTuple{N,Symbol}) where {N} =
    NamedTuple{s}(get_trace(t, x) for x in s)

"""
    get_trace(t::AbstractTrajectory, s::Symbol...)
"""
get_trace(t::AbstractTrajectory, s::Symbol...) = get_trace(t, s)

"""
    get_trace(t::AbstractTrajectory{names}) where {names}
"""
get_trace(t::AbstractTrajectory{names}) where {names} =
    NamedTuple{names}(get_trace(t, x) for x in names)

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

"""
    Base.push!(t::AbstractTrajectory; kwargs...)
"""
@interface function Base.push!(t::AbstractTrajectory; kwargs...)
    for kv in kwargs
        push!(t, kv)
    end
end

"""
    Base.push!(t::AbstractTrajectory, kv::Pair{Symbol})
"""
@interface Base.push!(t::AbstractTrajectory, kv::Pair{Symbol})

"""
    Base.pop!(t::AbstractTrajectory{names}) where {names}

`pop!` out one element of each trace in `t`
"""
@interface function Base.pop!(t::AbstractTrajectory{names}) where {names}
    pop!(t, names...)
end

"""
    Base.pop!(t::AbstractTrajectory, s::Symbol...)

`pop!` out one element of the traces specified in `s`
"""
@interface function Base.pop!(t::AbstractTrajectory, s::Symbol...)
    NamedTuple{s}(pop!(t, x) for x in s)
end

"""
    Base.pop!(t::AbstractTrajectory, s::Symbol)

`pop!` out one element of the trace `s` in `t`
"""
@interface Base.pop!(t::AbstractTrajectory, s::Symbol)

"""
    extract_experience(trajectory::AbstractTrajectory, learner::AbstractLearner)

Extract necessary data from the `trajectory` given a `learner`
"""
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
    (env::AbstractEnv)(action) = env(DEFAULT_PLAYER, action) -> nothing
    (env::AbstractEnv)(player, action) -> nothing

Super type of all reinforcement learning environments.
An environments is a functional object which takes in an action and returns `nothing`.

!!! note
    So why don't we adopt the `step!` method like OpenAI Gym here?
    The reason is that the async manner will simplify a lot of things here.
"""
@interface abstract type AbstractEnv end

abstract type AbstractDynamicStyle end

@interface struct Sequential <: AbstractDynamicStyle end

"Environment with the [`DynamicStyle`](@ref) of `SEQUENTIAL` must takes actions from different players one-by-one."
@interface const SEQUENTIAL = Sequential()
@interface struct Simultaneous <: AbstractDynamicStyle end

"Environment with the [`DynamicStyle`](@ref) of `SIMULTANEOUS` must take in actions from some (or all) players at one time"
@interface const SIMULTANEOUS = Simultaneous()

"""
    DynamicStyle(x::AbstractEnv) = SEQUENTIAL

Determine whether the players can play simultaneous or not. Default value is [`SEQUENTIAL`](@ref)
"""
@interface DynamicStyle(x::AbstractEnv) = SEQUENTIAL

@interface (env::AbstractEnv)(action) = env(DEFAULT_PLAYER, action)
@interface (env::AbstractEnv)(player, action)

"""
    get_action_space(env::AbstractEnv) -> AbstractSpace
"""
@interface get_action_space(env::AbstractEnv) = env.action_space

"""
    get_observation_space(env::AbstractEnv) -> AbstractSpace
"""
@interface get_observation_space(env::AbstractEnv) = env.observation_space

"Return [`DEFAULT_PLAYER`](@ref) by default"
@interface get_current_player(env::AbstractEnv) = DEFAULT_PLAYER

"""
    get_num_players(env::AbstractEnv) -> Int
"""
@interface get_num_players(env::AbstractEnv) = 1

"Show the environment in a user-friendly manner"
@interface render(::AbstractEnv)

"Reset the internal state of an environment"
@interface reset!(::AbstractEnv)

@interface Random.seed!(::AbstractEnv, seed)

#####
# Observation
#####

"""
    observe(env::AbstractEnv) = observe(env, get_current_player(env))
    observe(::AbstractEnv, player)

Get an observation of the `env` from the perspective of an `player`.

!!! note
    This is a very deliberate decision to adopt the duck-typing here
    to describe an observation from an environment.
    By default, we assume an observation is a NamedTuple,
    which is the most common case.
    But of course it can be of any type, as long as it implemented 
    the necessay methods described in this section.
"""
@interface observe(env::AbstractEnv) = observe(env, get_current_player(env))
@interface observe(::AbstractEnv, player)

"""
    get_legal_actions_mask(x) -> Bool[]

Only valid for observations of [`FULL_ACTION_SET`](@ref).
"""
@interface get_legal_actions_mask(x) = x.legal_actions_mask

"""
    get_legal_actions(x)

Only valid for observations of [`FULL_ACTION_SET`](@ref).
"""
@interface get_legal_actions(x) = findall(get_legal_actions_mask(x))

get_legal_actions(
    x::NamedTuple{(:reward, :terminal, :state, :legal_actions, :legal_actions_mask)},
) = x.legal_actions
get_legal_actions(x::NamedTuple{(:reward, :terminal, :state, :legal_actions)}) =
    x.legal_actions

"""
    get_terminal(x) -> bool
"""
@interface get_terminal(x) = x.terminal

"""
    get_reward(x) -> Number
"""
@interface get_reward(x) = x.reward

"""
    get_state(x) -> Array
"""
@interface get_state(x) = x.state

"By default [`INVALID_ACTION`](@ref) is returned"
@interface get_invalid_action(x) = INVALID_ACTION

#####
# Space
#####

"""
Describe the span of observations and actions.
Usually the following methods are implemented:

- `Base.length`
- `Base.in`
- `Random.rand`
- `Base.eltype`
"""
@interface abstract type AbstractSpace end

@interface Base.length(::AbstractSpace)
@interface Base.in(x, s::AbstractSpace)
@interface Random.rand(rng::AbstractRNG, s::AbstractSpace)
@interface Base.eltype(s::AbstractSpace)
