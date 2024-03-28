# ReinforcementLearningCore.jl

```@autodocs
Modules = [ReinforcementLearningCore]
```

In addition to containing the [run loop](./How_to_implement_a_new_algorithm.md), RLCore is a collection of pre-implemented components that are frequently used in RL.

## QBasedPolicy

[`QBasedPolicy`](@ref) is an [`AbstractPolicy`](@ref) that wraps a Q-Value _learner_ (tabular or approximated) and an _explorer_. Use this wrapper to implement a policy that directly uses a Q-value function to 
decide its next action. In that case, instead of creating an [`AbstractPolicy`](@ref) subtype for your algorithm, define an [`AbstractLearner`](@ref) subtype and specialize `RLBase.optimise!(::YourLearnerType, ::Stage, ::Trajectory)`. This way you will not have to code the interaction between your policy and the explorer yourself. 
RLCore provides the most common explorers (such as epsilon-greedy, UCB, etc.). You can find many examples of QBasedPolicies in the DQNs section of RLZoo.

## Parametric approximators 
### Approximator 

If your algorithm uses a neural network or a linear approximator to approximate a function trained with `Flux.jl`, use the `Approximator`. It 
wraps a `Flux` model and an `Optimiser` (such as Adam or SGD). Your `optimise!(::PolicyOrLearner, batch)` function will probably consist in computing a gradient 
and call the `RLBase.optimise!(app::Approximator, gradient::Flux.Grads)` after that. 

`Approximator` implements the `model(::Approximator)` and `target(::Approximator)` interface. Both return the underlying Flux model. The advantage of this interface is explained in the `TargetNetwork` section below.

### TargetNetwork

The use of a target network is frequent in state or action value-based RL. The principle is to hold a copy of of the main approximator, which is trained using a gradient, and a copy of it that is either only partially updated, or just less frequently updated. `TargetNetwork` is constructed by wrapping an `Approximator`. Set the `sync_freq` keyword argument to a value greater that one to copy the main model into the target every `sync_freq` updates, or set the `\rho` parameter to a value greater than 0 (usually 0.99f0) to let the target be partially updated towards the main model every update. `RLBase.optimise!(tn::TargetNetwork, gradient::Flux.Grads)` will take care of updating the target for you. 

The other advantage of `TargetNetwork` is that it uses Julia's multiple dispatch to let your algorithm be agnostic to the presence or absence of a target network. For example, the `DQNLearner` in RLZoo has an `approximator` field typed to be a `Union{Approximator, TargetNetwork}`. When computing the temporal difference error, the learner calls `Q = model(learner.approximator)` and `Qt = target(learner.approximator)`. If `learner.approximator` is a `Approximator`, then no target network is used because both calls point to the same neural network, if it is a `TargetNetwork` then the automatically managed target is returned. 

## Architectures

Common model architectures are also provided such as the `GaussianNetwork` for continuous policies with diagonal multivariate policies; and `CovGaussianNetwork` for full covariance (very slow on GPUs at the moment).
