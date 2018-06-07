# ReinforcementLearning

[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://jbrea.github.io/TabularReinforcementLearning.jl/latest)
[![Build Status](https://travis-ci.org/jbrea/TabularReinforcementLearning.jl.svg?branch=master)](https://travis-ci.org/jbrea/TabularReinforcementLearning.jl)
[![codecov](https://codecov.io/gh/jbrea/TabularReinforcementLearning.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/jbrea/TabularReinforcementLearning.jl)

A reinforcement learning library for [Julia](https://julialang.org/).


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
|Cartpole | 4D      | discrete     |
|Mountain Car | 2D  | discrete     |
|Pendulum | 3D     | 1D           |
|Atari       | pixel images | discrete|
|[POMDPs](https://github.com/JuliaPOMDP/POMDPs.jl), MDPs, Mazes, Cliffwalking | discrete | discrete|
|[OpenAi Gym](https://github.com/openai/gym) (with PyCall) | see [here](https://github.com/openai/gym) | see [here](https://github.com/openai/gym) |

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

    pkg.clone("https://github.com/jbrea/ReinforcementLearning.jl")


# Credits

- Main author: Johanni Brea
- Contributions: Marco Lehmann, Raphaël Nunes

# Contribute

Contributions are highly welcome. Please have a look at the issues.
