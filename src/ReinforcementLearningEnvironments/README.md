# ReinforcementLearningEnvironments.jl

[![Build Status](https://travis-ci.com/JuliaReinforcementLearning/ReinforcementLearningEnvironments.jl.svg?branch=master)](https://travis-ci.com/JuliaReinforcementLearning/ReinforcementLearningEnvironments.jl)

This package serves as a one-stop place for different kinds of reinforcement learning environments.

Install:

```julia
(v1.1) pkg> add https://github.com/JuliaReinforcementLearning/ReinforcementLearningEnvironments.jl
```

**TODO:**

- [x] Add a Docker file for quick test.
    ```
    $ docker run -it --rm juliareinforcementlearning/reinforcementlearningenvironments
    ```
- [ ] Add benchmarks.

## API

| Method | Description |
| :---  | :--------- |
| `observe(env;observer=:default)` | Return the observation of `env` from the view of `observer`|
| `reset!(env)` | Reset `env` to an initial state|
| `interact!(env, action)` | Send `action` to `env`. For multi-agent environments, `action` is a dictionary of actions from different agents|
| `action_space(env)` | Return the action space of `env` |
| `observation_space(env)` | Return the observation space of `env`|
| **Optional Methods** | |
| `render(env)` | Show the current state of environment |

## Supported Environments

By default, only some basic environments are installed. If you want to use some other environments, you'll need to add those dependencies correspondingly.

### Built-in Environments

- CartPoleEnv
- MountainCarEnv
- PendulumEnv
- MDPEnv
- POMDPEnv
- SimpleMDPEnv
  - deterministic_MDP
  - absorbing_deterministic_tree_MDP
  - stochastic_MDP
  - stochastic_tree_MDP
  - deterministic_tree_MDP_with_rand_reward
  - deterministic_tree_MDP
  - deterministic_MDP

### 3-rd Party Environments

| Environment Name | Dependent Package Name | Description |
| :--- | :--- | :--- |
| `AtariEnv` | [ArcadeLearningEnvironment.jl](https://github.com/JuliaReinforcementLearning/ArcadeLearningEnvironment.jl) | |
| `ViZDoomEnv` | [ViZDoom.jl](https://github.com/JuliaReinforcementLearning/ViZDoom.jl) | Currently only a basic environment is supported. (By calling `basic_ViZDoom_env()`)|
| `GymEnv` | [PyCall.jl](https://github.com/JuliaPy/PyCall.jl) | You need to manually install `gym` first |

**TODO:**

- [ ] [Hanabi](https://github.com/deepmind/hanabi-learning-environment) (Ongoing)
- [ ] Box2d (Investigating)
- [ ] Bullet (Investigating)

How to enable 3-rd party environments?

Take the `AtariEnv` for example:

1. Install this package by:
    ```julia
    (v1.1) pkg> add https://github.com/JuliaReinforcementLearning/ReinforcementLearningEnvironments.jl
    ```
2. Install corresponding dependent package by:
    ```julia
    (v1.1) pkg> add ArcadeLearningEnvironment
    ```
3. Using the above two packages:
    ```julia
    using ReinforcementLearningEnvironments
    using ArcadeLearningEnvironment
    env = AtariEnv("pong")
    ```

## Style Guide

We favor the [YASGuide](https://github.com/jrevels/YASGuide) style guide.
