# ReinforcementLearning

[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://JuliaReinforcementLearning.github.io/ReinforcementLearning.jl/latest)
[![Build Status](https://travis-ci.com/JuliaReinforcementLearning/ReinforcementLearning.jl.svg?branch=master)](https://travis-ci.com/JuliaReinforcementLearning/ReinforcementLearning.jl)
[![codecov](https://codecov.io/gh/JuliaReinforcementLearning/ReinforcementLearning.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaReinforcementLearning/ReinforcementLearning.jl)

A reinforcement learning package for [Julia](https://julialang.org/).


# What is reinforcement learning?

- [Wikipedia](https://en.wikipedia.org/wiki/Reinforcement_learning)
- [New Sutton & Barto book](http://incompleteideas.net/book/the-book-2nd.html)

# Features

## Learning methods

| name | discrete states | linear approximation | non-linear approximation |
|------|:---------------:|:--------------------:|:------------------------:|
|Q-learning/SARSA(λ) | ✓            |   ✓    |               | |
|n-step Q-learning/SARSA |✓            |   ✓                  |  |
|Online Policy Gradient |✓            |   ✓                  |  |
|Episodic Reinforce |✓            |   ✓                  |  |
|n-step Actor-Critic Policy-Gradient |✓            |   ✓                  |✓   |
|Monte Carlo Control |✓            |                  |  |
|Prioritized Sweeping|✓            |                    |  |
|(double) DQN |                                   |   ✓                  |✓   |


## Environments

|name | state space | action space |
|-----|-------------|--------------|
|[Cartpole](https://github.com/JuliaReinforcementLearning/ReinforcementLearningEnvironmentClassicControl.jl)| 4D      | discrete     |
|[Mountain Car](https://github.com/JuliaReinforcementLearning/ReinforcementLearningEnvironmentClassicControl.jl)| 2D  | discrete     |
|[Pendulum](https://github.com/JuliaReinforcementLearning/ReinforcementLearningEnvironmentClassicControl.jl) | 3D     | 1D           |
|[Atari](https://github.com/JuliaReinforcementLearning/ReinforcementLearningEnvironmentAtari.jl) | pixel images | discrete|
|[VizDoom](https://github.com/JuliaReinforcementLearning/ReinforcementLearningEnvironmentVizDoom.jl) | pixel images | discrete|
|[POMDPs](https://github.com/JuliaReinforcementLearning/ReinforcementLearningEnvironmentDiscrete.jl), MDPs, [Mazes](https://github.com/JuliaReinforcementLearning/ReinforcementLearningEnvironmentDiscrete.jl), [Cliffwalking](https://github.com/JuliaReinforcementLearning/ReinforcementLearningEnvironmentDiscrete.jl) | discrete | discrete|
|[OpenAi Gym](https://github.com/JuliaReinforcementLearning/ReinforcementLearningEnvironmentGym.jl) (using PyCall) | see [here](https://github.com/openai/gym) | see [here](https://github.com/openai/gym) |

## Preprocessors

- State Aggregation
- Tile Coding
- Random Projections
- Radial Basis Functions

## Helper Functions

- comparison of different methods
- callbacks to track performance, change exploration policy, save models during
  learning etc.

# Installation

```julia
(v1.0) pkg> add ReinforcementLearning
```
 or in julia v0.6

```julia
Pkg.add("ReinforcementLearning")
```

# Credits

- Main author: Johanni Brea
- Contributions: Marco Lehmann, Raphaël Nunes

# Contribute

Contributions are highly welcome. Please have a look at the issues.
