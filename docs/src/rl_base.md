# ReinforcementLearningBase.jl

```@docs
RLBase
```

## General

```@docs
AbstractStage
```

## Agent

```@docs
AbstractAgent
get_role
DEFAULT_PLAYER
```

## Policy

```@docs
AbstractPolicy
update!(p::AbstractPolicy, experience)
get_prob(p::AbstractPolicy, obs)
get_prob(p::AbstractPolicy, obs, a)
get_priority
```

## Learner

```@docs
AbstractLearner
update!(learner::AbstractLearner, experience)
get_priority(p::AbstractLearner, experience)
extract_experience(trajectory::AbstractTrajectory, learner::AbstractLearner)
```

## Approximator

```@docs
AbstractApproximator
batch_estimate
update!(a::AbstractApproximator, correction)
Base.copyto!(dest::AbstractApproximator, src::AbstractApproximator)
ApproximatorStyle
Q_APPROXIMATOR 
V_APPROXIMATOR
HYBRID_APPROXIMATOR 
```

## Explorer

```@docs
AbstractExplorer
reset!(p::AbstractExplorer)
Base.copy(p::AbstractExplorer)
get_prob(p::AbstractExplorer, x)
get_prob(p::AbstractExplorer, x, mask)
```

##  EnvironmentModel

```@docs
AbstractEnvironmentModel
```

## Environment

```@docs
AbstractEnv 
DynamicStyle
SIMULTANEOUS
SEQUENTIAL 
get_action_space
get_observation_space
get_current_player
get_num_players
render
reset!(::AbstractEnv)
```

Several meta-environments are also provided:

```@docs
WrappedEnv
```

## Preprocessor

```@docs
AbstractPreprocessor
```

## Observation

```@docs
observe
get_legal_actions_mask
get_legal_actions
get_terminal
get_reward
get_state
get_invalid_action
INVALID_ACTION
ActionStyle
MINIMAL_ACTION_SET
FULL_ACTION_SET
RTSA
SARTSA
```

Some meta-observations are also provided:

```@docs
StateOverriddenObs
```

## Trajectory

```@docs
AbstractTrajectory
get_trace
push!
pop!
```

## Space

```@docs
AbstractSpace
```

Some typical implementations of [`AbstractSpace`](@ref) are also included in the [`RLBase`](@ref).

```@docs
EmptySpace
DiscreteSpace
MultiDiscreteSpace
ContinuousSpace
MultiContinuousSpace
TupleSpace
DictSpace
```