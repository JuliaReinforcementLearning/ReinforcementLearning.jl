# ReinforcementLearningEnvironments.jl

[![Build Status](https://travis-ci.com/JuliaReinforcementLearning/ReinforcementLearningEnvironments.jl.svg?branch=master)](https://travis-ci.com/JuliaReinforcementLearning/ReinforcementLearningEnvironments.jl)

This package serves as a one-stop place for different kinds of reinforcement learning environments.

Install:

```julia
pkg> add ReinforcementLearningEnvironments
```

## API

All the environments here are supposed to have implemented the [`AbstractEnvironment`](https://github.com/JuliaReinforcementLearning/ReinforcementLearningBase.jl/blob/9205f6d7bdde5d17a5d2baedefcf8a1854b40698/src/interface.jl#L230-L261) related interfaces in [ReinforcementLearningBase.jl](https://github.com/JuliaReinforcementLearning/ReinforcementLearningBase.jl).

## Supported Environments

By default, only some basic environments are installed. If you want to use some other environments, you'll need to add those dependencies correspondingly.

### Built-in Environments

- AcrobotEnv
- CartPoleEnv
- MountainCarEnv
- ContinuousMountainCarEnv
- PendulumEnv

### 3-rd Party Environments

| Environment Name | Dependent Package Name | Description |
| :--- | :--- | :--- |
| `AtariEnv` | [ArcadeLearningEnvironment.jl](https://github.com/JuliaReinforcementLearning/ArcadeLearningEnvironment.jl) | Tested only on Linux|
| `ViZDoomEnv` | [ViZDoom.jl](https://github.com/JuliaReinforcementLearning/ViZDoom.jl) | Currently only a basic environment is supported. (By calling `basic_ViZDoom_env()`)|
| `GymEnv` | [PyCall.jl](https://github.com/JuliaPy/PyCall.jl) | Tested only on Linux |
| `MDPEnv`,`POMDPEnv`| [POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl)| The `get_observation_space` method is undefined|
| `OpenSpielEnv` | [OpenSpiel.jl](https://github.com/JuliaReinforcementLearning/OpenSpiel.jl) | |

## Usage

```julia
julia> using ReinforcementLearningEnvironments

julia> using ReinforcementLearningBase

julia> env = CartPoleEnv()
CartPoleEnv{Float64}(gravity=9.8,masscart=1.0,masspole=0.1,totalmass=1.1,halflength=0.5,polemasslength=0.05,forcemag=10.0,tau=0.02,thetathreshold=0.20943951023931953,xthreshold=2.4,max_steps=200)

julia> action_space = get_action_space(env)
DiscreteSpace{UnitRange{Int64}}(1:2)

julia> while true
           action = rand(action_space)
           env(action)
           obs = observe(env)
           get_terminal(obs) && break
       end
```
