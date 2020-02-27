<div align="center">
  <p>
  <img src="https://raw.githubusercontent.com/JuliaReinforcementLearning/ReinforcementLearning.jl/dev/docs/src/assets/logo.png">
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

```
+-------------------------------------------------------------------------------------------+
|                                                                                           |
|  ReinforcementLearning.jl                                                                 |
|                                                                                           |
|      +------------------------------+                                                     |
|      | ReinforcementLearningBase.jl |                                                     |
|      +--------|---------------------+                                                     |
|               |                                                                           |
|               |         +--------------------------------------+                          |
|               |         | ReinforcementLearningEnvironments.jl |                          |
|               |         |                                      |                          |
|               |         |     (Conditionally depends on)       |                          |
|               |         |                                      |                          |
|               |         |     ArcadeLearningEnvironment.jl     |                          |
|               +-------->+     OpenSpiel.jl                     |                          |
|               |         |     POMDPs.jl                        |                          |
|               |         |     PyCall.jl                        |                          |
|               |         |     ViZDoom.jl                       |                          |
|               |         |     Maze.jl(WIP)                     |                          |
|               |         +--------------------------------------+                          |
|               |                                                                           |
|               |         +------------------------------+                                  |
|               +-------->+ ReinforcementLearningCore.jl |                                  |
|                         +--------|---------------------+                                  |
|                                  |                                                        |
|                                  |          +-----------------------------+               |
|                                  |--------->+ ReinforcementLearningZoo.jl |               |
|                                  |          +-----------------------------+               |
|                                  |                                                        |
|                                  |          +----------------------------------------+    |
|                                  +--------->+ ReinforcementLearningAnIntroduction.jl |    |
|                                             +----------------------------------------+    |
+-------------------------------------------------------------------------------------------+
```

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

