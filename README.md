<div align="center">
  <p>
  <img src="https://raw.githubusercontent.com/JuliaReinforcementLearning/ReinforcementLearning.jl/master/docs/src/assets/logo.png" width="320">
  </p>

  <p>
  <a href="https://JuliaReinforcementLearning.github.io/ReinforcementLearning.jl/latest"><img src="https://img.shields.io/badge/docs-latest-blue.svg"></a>
  <a href="https://travis-ci.com/JuliaReinforcementLearning/ReinforcementLearning.jl"><img src="https://travis-ci.com/JuliaReinforcementLearning/ReinforcementLearning.jl.svg?branch=master"></a>
  </p>
</div>

**ReinforcementLearning.jl**, as the name says, is a package for reinforcement learning research in Julia.

Our design principles are:

- **Reusability and extensibility**: Provide elaborately designed components and interfaces to help users implement new algorithms.
- **Easy experimentation**: Make it easy for new users to run benchmark experiments, compare different algorithms, evaluate and diagnose agents.
- **Reproducibility**: Facilitate reproducibility from traditional tabular methods to modern deep reinforcement learning algorithms.

## Project Structure

`ReinforcementLearning.jl` itself is just a wrapper around several other packages inside the [JuliaReinforcementLearning](https://github.com/JuliaReinforcementLearning) org. The relationship between different packages is described below:

<pre>+-------------------------------------------------------------------------------------------+
|                                                                                           |
|  <a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl">ReinforcementLearning.jl</a>                                                                 |
|                                                                                           |
|      +------------------------------+                                                     |
|      | <a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearningBase.jl">ReinforcementLearningBase.jl</a> |                                                     |
|      +--------|---------------------+                                                     |
|               |                                                                           |
|               |         +--------------------------------------+                          |
|               |         | <a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearningEnvironments.jl">ReinforcementLearningEnvironments.jl</a> |                          |
|               |         |                                      |                          |
|               |         |     (Conditionally depends on)       |                          |
|               |         |                                      |                          |
|               |         |     <a href="https://github.com/JuliaReinforcementLearning/ArcadeLearningEnvironment.jl">ArcadeLearningEnvironment.jl</a>     |                          |
|               +--------&gt;+     <a href="https://github.com/JuliaReinforcementLearning/OpenSpiel.jl">OpenSpiel.jl</a>                     |                          |
|               |         |     <a href="https://github.com/JuliaPOMDP/POMDPs.jl">POMDPs.jl</a>                        |                          |
|               |         |     <a href="https://github.com/JuliaPy/PyCall.jl">PyCall.jl</a>                        |                          |
|               |         |     <a href="https://github.com/JuliaReinforcementLearning/ViZDoom.jl">ViZDoom.jl</a>                       |                          |
|               |         |     Maze.jl(WIP)                     |                          |
|               |         +--------------------------------------+                          |
|               |                                                                           |
|               |         +------------------------------+                                  |
|               +--------&gt;+ <a href="">ReinforcementLearningCore.jl</a> |                                  |
|                         +--------|---------------------+                                  |
|                                  |                                                        |
|                                  |          +-----------------------------+               |
|                                  |---------&gt;+ <a href="https://github.com/JuliaReinforcementLearning/ViZDoom.jl">ReinforcementLearningZoo.jl</a> |               |
|                                  |          +-----------------------------+               |
|                                  |                                                        |
|                                  |          +----------------------------------------+    |
|                                  +---------&gt;+ <a href="https://github.com/JuliaReinforcementLearning/ReinforcementLearningAnIntroduction.jl">ReinforcementLearningAnIntroduction.jl</a> |    |
|                                             +----------------------------------------+    |
+-------------------------------------------------------------------------------------------+
</pre>

Key capabilities/features include:

- Well tested traditional methods:
    - `TDLearner`
    - `DifferentialTDLearner`
    - `TDλReturnLearner`
    - `DoubleLearner`
    - `MonteCarloLearner`
    - `GradientBanditLearner`
    - `ReinforcePolicy`

- Efficiently implemented deep reinforcement learning algorithms:
    - Deep Q-Learning:
        - `BasicDQNLearner`
        - `DQNLearner`
        - `PrioritizedDQNLearner`
        - `RainbowLearner`

## Installation

This package can be installed from the package manager in Julia's REPL:

```
] add ReinforcementLearning
```

## A Quick Example

