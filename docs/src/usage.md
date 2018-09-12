## Simple usage

1. Choose a [learner](@ref learners).
2. Choose an [environment](@ref environments).
3. Choose a [stopping criterion](@ref stop).
4. (Optionally) choose [callbacks](@ref).
5. (Optionally) choose a [preprocessor](@ref preprocessors).
6. Define an [`RLSetup`](@ref).
7. Learn with [`learn!`](@ref).
8. Look at results with [`getvalue`](@ref getvalue).

### Example 1

```julia
using ReinforcementLearning, ReinforcementLearningEnvironmentDiscrete

learner = QLearning()
env = MDP()
stop = ConstantNumberSteps(10^3)
x = RLSetup(learner, env, stop, callbacks = [TotalReward()])
learn!(x)
getvalue(x.callbacks[1])
```

### Example 2

```julia
using ReinforcementLearning, ReinforcementLearningEnvironmentClassicControl, Flux

learner = DQN(Chain(Dense(4, 24, relu), Dense(24, 48, relu), Dense(48, 2)),
              opttype = x -> ADAM(x, .001))
env = CartPole()
stop = ConstantNumberEpisodes(2*10^3)
callbacks = [EvaluateGreedy(EvaluationPerEpisode(TimeSteps(), returnmean=true),
                            ConstantNumberEpisodes(100), every = Episode(100)),
             EvaluationPerEpisode(TimeSteps()),
             Progress()]
x = RLSetup(learner, env, stop, callbacks = callbacks)
learn!(x)
getvalue(x.callbacks[1])
```


## Comparisons

See section [`Comparison`](@ref comparison).

## Examples

See [environments](@ref environments)
