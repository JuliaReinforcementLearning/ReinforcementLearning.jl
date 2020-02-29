# ReinforcementLearningCore.jl

```@docs
RLCore
```

## Core

The most important function in RLCore is `run(agent, env, stop_condition, hook)`.
In practice, it will be dispatched to different implementations based on the type
of `agent` and `env`. For a `stop_condition`, it can be arbitrary function which
accepts `agent, env, obs` as arguments and return a `Bool` value indicates whether
to stop or not. For a `hook`, it can should be instances of [`AbstractHook`](@ref).

### Stop Conditions

```@docs
ComposedStopCondition
StopAfterStep
StopAfterEpisode
StopWhenDone
```

### Hooks

```@docs
AbstractHook
EmptyHook
ComposedHook
StepsPerEpisode
RewardsPerEpisode
TotalRewardPerEpisode
CumulativeReward
TimePerStep
```

## Agents

```@docs
Agent
DynaAgent
```

## Policies

```@docs
OffPolicy
QBasedPolicy
RandomPolicy
VBasedPolicy
WeightedRandomPolicy
```

## Trajectories

```@docs
Trajectory
VectorialTrajectory
CircularTrajectory
VectorialCompactSARTSATrajectory
EpisodicCompactSARTSATrajectory
CircularCompactSARTSATrajectory
CircularCompactPSARTSATrajectory
```

## Preprocessors

```@docs
ComposedPreprocessor
CloneStatePreprocessor
```

## Learners

```@docs
DoubleLearner
```

## Approximators

```@docs
TabularApproximator
NeuralNetworkApproximator
```

## Explorers

```@docs
EpsilonGreedyExplorer
UCBExplorer
WeightedExplorer
```

## Utils

```@docs
huber_loss
huber_loss_unreduced
CircularArrayBuffer
device
SumTree
find_all_max
```