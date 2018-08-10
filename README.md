# ReinforcementLearning

[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://jbrea.github.io/ReinforcementLearning.jl/latest)
[![Build Status](https://travis-ci.org/jbrea/ReinforcementLearning.jl.svg?branch=master)](https://travis-ci.org/jbrea/ReinforcementLearning.jl)
[![codecov](https://codecov.io/gh/jbrea/ReinforcementLearning.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/jbrea/ReinforcementLearning.jl)

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
|[Cartpole](https://github.com/JuliaReinforcementLearning/RLEnvClassicControl.jl)| 4D      | discrete     |
|[Mountain Car](https://github.com/JuliaReinforcementLearning/RLEnvClassicControl.jl)| 2D  | discrete     |
|[Pendulum](https://github.com/JuliaReinforcementLearning/RLEnvClassicControl.jl) | 3D     | 1D           |
|[Atari](https://github.com/JuliaReinforcementLearning/RLEnvAtari.jl) | pixel images | discrete|
|[VizDoom](https://github.com/JuliaReinforcementLearning/RLEnvVizDoom.jl) | pixel images | discrete|
|[POMDPs](https://github.com/JuliaReinforcementLearning/RLEnvDiscrete.jl), MDPs, [Mazes](https://github.com/JuliaReinforcementLearning/RLEnvDiscrete.jl), [Cliffwalking](https://github.com/JuliaReinforcementLearning/RLEnvDiscrete.jl) | discrete | discrete|
|[OpenAi Gym](https://github.com/JuliaReinforcementLearning/RLEnvGym.jl) (using PyCall) | see [here](https://github.com/openai/gym) | see [here](https://github.com/openai/gym) |

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
Pkg.add("ReinforcementLearning")
```

# Credits

- Main author: Johanni Brea
- Contributions: Marco Lehmann, Raphaël Nunes

# Contribute

Contributions are highly welcome. Please have a look at the issues.
