```@raw html
<div align="center">
  <p>
  <img src="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/raw/master/docs/src/assets/logo.svg?sanitize=true" width="320px">
  </p>

  <p>
  <a href="https://JuliaReinforcementLearning.github.io/ReinforcementLearning.jl/latest"><img src="https://img.shields.io/badge/docs-latest-blue.svg"></a>
  <a href="https://travis-ci.com/JuliaReinforcementLearning/ReinforcementLearning.jl"><img src="https://travis-ci.com/JuliaReinforcementLearning/ReinforcementLearning.jl.svg?branch=master"></a>
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

```@raw html
<pre>+-----------------------------------------------------------------------------------+
|                                                                                   |
|  <a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl">ReinforcementLearning.jl</a>                                                         |
|                                                                                   |
|      +------------------------------+                                             |
|      | <a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearningBase.jl">ReinforcementLearningBase.jl</a> |                                             |
|      +----|-------------------------+                                             |
|           |                                                                       |
|           |     +--------------------------------------+                          |
|           +----&gt;+ <a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearningEnvironments.jl">ReinforcementLearningEnvironments.jl</a> |                          |
|           |     +--------------------------------------+                          |
|           |                                                                       |
|           |     +------------------------------+                                  |
|           +----&gt;+ <a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearningCore.jl">ReinforcementLearningCore.jl</a> |                                  |
|                 +----|-------------------------+                                  |
|                      |                                                            |
|                      |     +-----------------------------+                        |
|                      +----&gt;+ <a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearningZoo.jl">ReinforcementLearningZoo.jl</a> |                        |
|                            +----|------------------------+                        |
|                                 |                                                 |
|                                 |     +-------------------------------------+     |
|                                 +----&gt;+ <a href="https://github.com/JuliaReinforcementLearning/DistributedReinforcementLearning.jl">DistributedReinforcementLearning.jl</a> |     |
|                                       +-------------------------------------+     |
|                                                                                   |
+-----------------------------------------------------------------------------------+
</pre>
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
