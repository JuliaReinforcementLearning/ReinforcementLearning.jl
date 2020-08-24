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
- PendulumNonInteractiveEnv

### 3-rd Party Environments

| Environment Name | Dependent Package Name | Description |
| :--- | :--- | :--- |
| `AtariEnv` | [ArcadeLearningEnvironment.jl](https://github.com/JuliaReinforcementLearning/ArcadeLearningEnvironment.jl) | |
| `ViZDoomEnv` | [ViZDoom.jl](https://github.com/JuliaReinforcementLearning/ViZDoom.jl) | Broken [help wanted](https://github.com/JuliaReinforcementLearning/ViZDoom.jl/issues/7) |
| `GymEnv` | [PyCall.jl](https://github.com/JuliaPy/PyCall.jl) | |
| `MDPEnv`,`POMDPEnv`| [POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl)| Tested with `POMDPs.jl@v0.9`|
| `OpenSpielEnv` | [OpenSpiel.jl](https://github.com/JuliaReinforcementLearning/OpenSpiel.jl) | |
| `SnakeGameEnv` | [SnakeGames.jl](https://github.com/JuliaReinforcementLearning/SnakeGames.jl) | `SingleAgent`/`Multi-Agent`, `FullActionSet`/`MinimalActionSet`|

## Usage

```julia
julia> using ReinforcementLearningEnvironments

julia> using ReinforcementLearningBase

julia> env = CartPoleEnv()
# CartPoleEnv

## Traits

| Trait Type       |                Value |
|:---------------- | --------------------:|
| NumAgentStyle    |        SingleAgent() |
| DynamicStyle     |         Sequential() |
| InformationStyle | PerfectInformation() |
| ChanceStyle      |      Deterministic() |
| RewardStyle      |         StepReward() |
| UtilityStyle     |         GeneralSum() |
| ActionStyle      |   MinimalActionSet() |

## Actions

DiscreteSpace{UnitRange{Int64}}(1:2)

## Players

  * `DEFAULT_PLAYER`

## Current Player

`DEFAULT_PLAYER`

## Is Environment Terminated?

No

julia> get_state(env)
4-element Array{Float64,1}:
  0.02688439956517477
 -0.0003235577964125977
  0.019563124862911535
 -0.01897808522860225

julia> actions = get_actions(env)
DiscreteSpace{UnitRange{Int64}}(1:2)

julia> while true
           env(rand(actions))
           get_terminal(env) && break
       end
```

## Application

Checkout [atari.jl](https://github.com/JuliaReinforcementLearning/ReinforcementLearningZoo.jl/blob/master/src/experiments/atari.jl) for some more complicated cases on how to use these environments and the [wrappers](https://github.com/JuliaReinforcementLearning/ReinforcementLearningBase.jl/blob/master/src/implementations/environments.jl) provided in [ReinforcementLearningBase.jl](https://github.com/JuliaReinforcementLearning/ReinforcementLearningBase.jl).