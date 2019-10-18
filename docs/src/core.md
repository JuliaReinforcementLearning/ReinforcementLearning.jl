# Core

```@docs
run
policy_evaluation!
policy_improvement!
policy_iteration!
value_iteration!
```


## Stop Conditions

```@docs
StopAfterStep
StopAfterEpisode
StopWhenDone
ComposedStopCondition
```

## Hooks

```@docs
AbstractHook
ComposedHook
EmptyHook
StepsPerEpisode
RewardsPerEpisode
TotalRewardPerEpisode
CumulativeReward
TimePerStep
```