# ReinforcementLearningCore.jl

```@docs
RLCore
```

## Core

The most important function in RLCore is `run(agent, env, stop_condition, hook)`.
In practice, it will be dispatched to different implementations based on the type
of `agent` and `env`. For a `stop_condition`, it can be arbitrary function which
accepts `agent, env, obs` as arguments and return a `Bool` value indicates whether
to stop or not.

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
BatchStepsPerEpisode
RewardsPerEpisode
TotalRewardPerEpisode
TotalBatchRewardPerEpisode
CumulativeReward
TimePerStep
DoEveryNStep
DoEveryNEpisode
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
VBasedPolicy
```

## Learners

```@docs
AbstractLearner 
```

## Approximators

```@docs
AbstractApproximator
TabularApproximator
NeuralNetworkApproximator
ActorCritic
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
ResizeImage
StackFrames
```

## Explorers

```@docs
BatchExplorer
EpsilonGreedyExplorer
UCBExplorer
GumbelSoftmaxExplorer
WeightedExplorer
```

## Utils

```@docs
huber_loss
huber_loss_unreduced
generalized_advantage_estimation
CircularArrayBuffer
device
SumTree
find_all_max
```