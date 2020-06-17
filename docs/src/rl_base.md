# ReinforcementLearningBase.jl

```@docs
RLBase
```

## Policy

```@docs
AbstractPolicy
update!(p::AbstractPolicy, experience)
get_prob(p::AbstractPolicy, obs)
get_prob(p::AbstractPolicy, obs, a)
get_priority(π::AbstractPolicy, experience)
```

##  EnvironmentModel

```@docs
AbstractEnvironmentModel
```

## Environment

```@docs
AbstractEnv 
get_action_space
get_state_space
get_current_player
get_player_id
get_num_players
reset!(::AbstractEnv)
render
Random.seed!(env::AbstractEnv, seed)
Base.copy(env::AbstractEnv)
get_history(env::AbstractEnv)
```

## Traits of Environment

```@docs
DynamicStyle
ChanceStyle
InformationStyle
RewardStyle
UtilityStyle
ActionStyle

SEQUENTIAL
SIMULTANEOUS
DETERMINISTIC 
STOCHASTIC 
PERFECT_INFORMATION
IMPERFECT_INFORMATION
ZERO_SUM
CONSTANT_SUM
GENERAL_SUM
IDENTICAL_REWARD
FULL_ACTION_SET 
MINIMAL_ACTION_SET 
```

Several meta-environments are also provided:

```@docs
WrappedEnv
```

## Observation

```@docs
observe
get_legal_actions_mask
get_legal_actions
get_terminal
get_reward
get_state
```

## Space

```@docs
AbstractSpace
EmptySpace
DiscreteSpace
MultiDiscreteSpace
ContinuousSpace
MultiContinuousSpace
TupleSpace
DictSpace
```

## Utils

```
Base.run(π, env::AbstractEnv)
RandomPolicy
StateOverriddenObs
BatchObs
WrappedEnv
MultiThreadEnv
AbstractPreprocessor
CloneStatePreprocessor
ComposedPreprocessor
```