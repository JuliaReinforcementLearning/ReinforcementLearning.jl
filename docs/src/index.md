```@raw html
<div align="center">
  <p>
  <img src="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/raw/master/docs/src/assets/logo.svg?sanitize=true" width="320px">
  </p>

  <p>
  <a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/actions?query=workflow%3ACI"><img src="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/workflows/CI/badge.svg"></a>
  <a href="https://juliahub.com/ui/Packages/ReinforcementLearning/6l2TO"><img src="https://juliahub.com/docs/ReinforcementLearning/pkgeval.svg"></a>
  <a href="https://juliahub.com/ui/Packages/ReinforcementLearning/6l2TO"><img src="https://juliahub.com/docs/ReinforcementLearning/version.svg"></a>
  <a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/blob/master/LICENSE.md"><img src="http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat"></a>
  <a href="https://julialang.org/slack/"><img src=https://img.shields.io/badge/Chat%20on%20Slack-%23reinforcement--learnin-ff69b4"></a>
  <a href="https://github.com/SciML/ColPrac"><img src="https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet"></a>
  </p>
</div>
```

[**ReinforcementLearning.jl**](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl),
as the name says, is a package for reinforcement learning research in Julia.

Our design principles are:

- **Reusability and extensibility**: Provide elaborately designed components and
  interfaces to help users implement new algorithms.
- **Easy experimentation**: Make it easy for new users to run benchmark
  experiments, compare different algorithms, evaluate and diagnose agents.
- **Reproducibility**: Facilitate reproducibility from traditional tabular
  methods to modern deep reinforcement learning algorithms.

## Get Started

```julia
julia> ] add ReinforcementLearning

julia> using ReinforcementLearning

julia> run(E`JuliaRL_BasicDQN_CartPole`)
```

Check out the [Get Started](https://juliareinforcementlearning.org/get_started/) page for more detailed explanation!

## Project Structure

`ReinforcementLearning.jl` itself is just a wrapper around several other packages inside the [JuliaReinforcementLearning](https://github.com/JuliaReinforcementLearning) org. The relationship between different packages is described below:

```
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  ReinforcementLearning.jl                                                         |
|                                                                                   |
|      +------------------------------+                                             |
|      | ReinforcementLearningBase.jl |                                             |
|      +----|-------------------------+                                             |
|           |                                                                       |
|           |     +--------------------------------------+                          |
|           +---->+ ReinforcementLearningEnvironments.jl |                          |
|           |     +--------------------------------------+                          |
|           |                                                                       |
|           |     +------------------------------+                                  |
|           +---->+ ReinforcementLearningCore.jl |                                  |
|                 +----|-------------------------+                                  |
|                      |                                                            |
|                      |     +-----------------------------+                        |
|                      +---->+ ReinforcementLearningZoo.jl |                        |
|                            +----|------------------------+                        |
|                                 |                                                 |
|                                 |     +-------------------------------------+     |
|                                 +---->+ DistributedReinforcementLearning.jl |     |
|                                       +-------------------------------------+     |
|                                                                                   |
+-----------------------------------------------------------------------------------+
```

### Scope of Each Package

- [ReinforcementLearningBase.jl](https://github.com/JuliaReinforcementLearning/ReinforcementLearningBase.jl)
  Two main concepts in reinforcement learning are precisely defined here: **Policy**
  and **Environment**.
- [ReinforcementLearningEnvironments.jl](https://github.com/JuliaReinforcementLearning/ReinforcementLearningEnvironments.jl)
  Typical environment examples in pure Julia and wrappers for 3-rd party
  environments are provided in this package.
- [ReinforcementLearningCore.jl](https://github.com/JuliaReinforcementLearning/ReinforcementLearningCore.jl)
  Common utility functions and different layers of abstractions are contained in
  this package.
- [ReinforcementLearningZoo.jl](https://github.com/JuliaReinforcementLearning/ReinforcementLearningZoo.jl)
  Common reinforcement learning algorithms and their typical applications (aka
  `Experiment`s) are collected in this package.
- [DistributedReinforcementLearning.jl](https://github.com/JuliaReinforcementLearning/DistributedReinforcementLearning.jl)
  This package is still experimental and is not included in
  `ReinforcementLearning.jl` yet. Its goal is to extend some algorithms in
  `ReinforcementLearningZoo.jl` to apply them in distributed computing systems.

## Supporting 🖖

`ReinforcementLearning.jl` is a MIT licensed open source project with its
ongoing development made possible by many contributors in their spare time.
However, modern reinforcement learning research requires huge computing
resource, which is unaffordable for individual contributors. So if you or your
organization could provide the computing resource in some degree and would like
to cooperate in some way, please contact us!

## Citing

If you use `ReinforcementLearning.jl` in a scientific publication, we would
appreciate references to the following BibTex entry:

```
@misc{Tian2020Reinforcement,
  author       = {Jun Tian and other contributors},
  title        = {ReinforcementLearning.jl: A Reinforcement Learning Package for the Julia Language},
  year         = 2020,
  url          = {https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl}
}
```