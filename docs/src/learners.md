# [Learners](@id learners)

## TD Learner
```@autodocs
Modules = [ReinforcementLearning]
Pages   = ["tdlearning.jl", "traces.jl"]
```
### [Initial values, novel actions and unseen values](@id initunseen)
For td-error dependent methods, the exploration-exploitation trade-off depends
on the `initvalue` and the `unseenvalue`.  To distinguish actions that were
never choosen before, i.e. novel actions, the default initial Q-value (field
`param`) is `initvalue = Inf64`. In a state with novel actions, the
[policy](@ref policies) determines how to deal with novel actions. To compute
the td-error the `unseenvalue` is used for states with novel actions.  One way
to achieve agressively exploratory behavior is to assure that `unseenvalue` (or
`initvalue`) is larger than the largest possible Q-value.

## Policy Gradient Learner
```@autodocs
Modules = [ReinforcementLearning]
Pages   = ["policygradientlearning.jl"]
```

## N-step Learner
```@autodocs
Modules = [ReinforcementLearning]
Pages   = ["montecarlo.jl"]
```

## Model Based Learner
```@autodocs
Modules = [ReinforcementLearning]
Pages   = ["mdplearner.jl", "prioritizedsweeping.jl"]
```

## Deep Reinforcement Learning
```@autodocs
Modules = [ReinforcementLearning]
Pages   = ["dqn.jl", "deepactorcritic.jl"]
```

