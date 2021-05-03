# ReinforcementLearningEnvironments.jl

![CI](https://github.com/JuliaReinforcementLearning/ReinforcementLearningEnvironments.jl/workflows/CI/badge.svg)

This package serves as a one-stop place for different kinds of reinforcement learning environments.

Install:

```julia
pkg> add ReinforcementLearningEnvironments
```

## API

All the environments here are supposed to have implemented the [`AbstractEnv`](https://github.com/JuliaReinforcementLearning/ReinforcementLearningBase.jl/blob/9205f6d7bdde5d17a5d2baedefcf8a1854b40698/src/interface.jl#L230-L261) related interfaces in [ReinforcementLearningBase.jl](https://github.com/JuliaReinforcementLearning/ReinforcementLearningBase.jl).

## Supported Environments

By default, only some basic environments are installed and exported to make sure
that they work on different kinds of platforms. If you want to use some other
environments, you'll need to add those dependencies correspondingly.

### Built-in Environments

<table>
<th colspan="2">Traits</th><th> 1 </th><th> 2 </th><th> 3 </th><th> 4 </th><th> 5 </th><th> 6 </th><th> 7 </th><th> 8 </th><th> 9 </th><th> 10 </th><th> 11 </th><th> 12 </th><th> 13 </th><tr> <th rowspan="2"> ActionStyle </th><th> MinimalActionSet </th><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> </td> <td> ✔ </td><td> </td> <td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td></tr>
<tr> <th> FullActionSet </th><td> </td> <td> </td> <td> </td> <td> ✔ </td><td> </td> <td> ✔ </td><td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> </td> </tr>
<tr> <th rowspan="3"> ChanceStyle </th><th> Stochastic </th><td> ✔ </td><td> </td> <td> ✔ </td><td> ✔ </td><td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td></tr>
<tr> <th> Deterministic </th><td> </td> <td> ✔ </td><td> </td> <td> </td> <td> ✔ </td><td> ✔ </td><td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> </td> </tr>
<tr> <th> ExplicitStochastic </th><td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> ✔ </td><td> ✔ </td><td> ✔ </td><td> </td> <td> </td> <td> </td> <td> </td> </tr>
<tr> <th rowspan="2"> DefaultStateStyle </th><th> Observation </th><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> </td> <td> ✔ </td><td> </td> <td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td></tr>
<tr> <th> InformationSet </th><td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> ✔ </td><td> </td> <td> ✔ </td><td> </td> <td> </td> <td> </td> <td> </td> </tr>
<tr> <th rowspan="2"> DynamicStyle </th><th> Simultaneous </th><td> </td> <td> </td> <td> </td> <td> </td> <td> ✔ </td><td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> </td> </tr>
<tr> <th> Sequential </th><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> </td> <td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td></tr>
<tr> <th rowspan="2"> InformationStyle </th><th> PerfectInformation </th><td> </td> <td> ✔ </td><td> </td> <td> </td> <td> </td> <td> ✔ </td><td> </td> <td> ✔ </td><td> </td> <td> </td> <td> </td> <td> </td> <td> </td> </tr>
<tr> <th> ImperfectInformation </th><td> ✔ </td><td> </td> <td> ✔ </td><td> ✔ </td><td> ✔ </td><td> </td> <td> ✔ </td><td> </td> <td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td></tr>
<tr> <th rowspan="2"> NumAgentStyle </th><th> MultiAgent </th><td> </td> <td> </td> <td> </td> <td> </td> <td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> </td> <td> </td> <td> </td> <td> </td> </tr>
<tr> <th> SingleAgent </th><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td></tr>
<tr> <th rowspan="2"> RewardStyle </th><th> TerminalReward </th><td> ✔ </td><td> ✔ </td><td> </td> <td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> </td> <td> </td> <td> </td> <td> </td> </tr>
<tr> <th> StepReward </th><td> </td> <td> </td> <td> ✔ </td><td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td></tr>
<tr> <th rowspan="3"> StateStyle </th><th> Observation </th><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> </td> <td> ✔ </td><td> </td> <td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td></tr>
<tr> <th> InformationSet </th><td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> ✔ </td><td> </td> <td> ✔ </td><td> </td> <td> </td> <td> </td> <td> </td> </tr>
<tr> <th> InternalState </th><td> </td> <td> </td> <td> ✔ </td><td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> </td> </tr>
<tr> <th rowspan="4"> UtilityStyle </th><th> GeneralSum </th><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td><td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> ✔ </td><td> ✔ </td><td> ✔ </td><td> ✔ </td></tr>
<tr> <th> ZeroSum </th><td> </td> <td> </td> <td> </td> <td> </td> <td> ✔ </td><td> ✔ </td><td> </td> <td> </td> <td> ✔ </td><td> </td> <td> </td> <td> </td> <td> </td> </tr>
<tr> <th> ConstantSum </th><td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> ✔ </td><td> </td> <td> </td> <td> </td> <td> </td> <td> </td> </tr>
<tr> <th> IdenticalUtility </th><td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> ✔ </td><td> </td> <td> </td> <td> </td> <td> </td> <td> </td> <td> </td> </tr>
</table>
<ol><li> <a href="https://juliareinforcementlearning.org/ReinforcementLearning.jl/latest/rl_envs/#ReinforcementLearningEnvironments.MultiArmBanditsEnv-Tuple{}"> MultiArmBanditsEnv </a></li>
<li> <a href="https://juliareinforcementlearning.org/ReinforcementLearning.jl/latest/rl_envs/#ReinforcementLearningEnvironments.RandomWalk1D-Tuple{}"> RandomWalk1D </a></li>
<li> <a href="https://juliareinforcementlearning.org/ReinforcementLearning.jl/latest/rl_envs/#ReinforcementLearningEnvironments.TigerProblemEnv-Tuple{}"> TigerProblemEnv </a></li>
<li> <a href="https://juliareinforcementlearning.org/ReinforcementLearning.jl/latest/rl_envs/#ReinforcementLearningEnvironments.MontyHallEnv-Tuple{}"> MontyHallEnv </a></li>
<li> <a href="https://juliareinforcementlearning.org/ReinforcementLearning.jl/latest/rl_envs/#ReinforcementLearningEnvironments.RockPaperScissorsEnv-Tuple{}"> RockPaperScissorsEnv </a></li>
<li> <a href="https://juliareinforcementlearning.org/ReinforcementLearning.jl/latest/rl_envs/#ReinforcementLearningEnvironments.TicTacToeEnv-Tuple{}"> TicTacToeEnv </a></li>
<li> <a href="https://juliareinforcementlearning.org/ReinforcementLearning.jl/latest/rl_envs/#ReinforcementLearningEnvironments.TinyHanabiEnv-Tuple{}"> TinyHanabiEnv </a></li>
<li> <a href="https://juliareinforcementlearning.org/ReinforcementLearning.jl/latest/rl_envs/#ReinforcementLearningEnvironments.PigEnv-Tuple{}"> PigEnv </a></li>
<li> <a href="https://juliareinforcementlearning.org/ReinforcementLearning.jl/latest/rl_envs/#ReinforcementLearningEnvironments.KuhnPokerEnv-Tuple{}"> KuhnPokerEnv </a></li>
<li> <a href="https://juliareinforcementlearning.org/ReinforcementLearning.jl/latest/rl_envs/#ReinforcementLearningEnvironments.AcrobotEnv-Tuple{}"> AcrobotEnv </a></li>
<li> <a href="https://juliareinforcementlearning.org/ReinforcementLearning.jl/latest/rl_envs/#ReinforcementLearningEnvironments.CartPoleEnv-Tuple{}"> CartPoleEnv </a></li>
<li> <a href="https://juliareinforcementlearning.org/ReinforcementLearning.jl/latest/rl_envs/#ReinforcementLearningEnvironments.MountainCarEnv-Tuple{}"> MountainCarEnv </a></li>
<li> <a href="https://juliareinforcementlearning.org/ReinforcementLearning.jl/latest/rl_envs/#ReinforcementLearningEnvironments.PendulumEnv-Tuple{}"> PendulumEnv </a></li>
</ol>

**Note**: Many traits are *borrowed* from [OpenSpiel](https://github.com/deepmind/open_spiel).

### 3-rd Party Environments

| Environment Name | Dependent Package Name | Description |
| :--- | :--- | :--- |
| `AtariEnv` | [ArcadeLearningEnvironment.jl](https://github.com/JuliaReinforcementLearning/ArcadeLearningEnvironment.jl) | |
| `GymEnv` | [PyCall.jl](https://github.com/JuliaPy/PyCall.jl) | |
| `OpenSpielEnv` | [OpenSpiel.jl](https://github.com/JuliaReinforcementLearning/OpenSpiel.jl) | |
| `SnakeGameEnv` | [SnakeGames.jl](https://github.com/JuliaReinforcementLearning/SnakeGames.jl) | `SingleAgent`/`Multi-Agent`, `FullActionSet`/`MinimalActionSet`|
| [#list-of-environments](https://github.com/JuliaReinforcementLearning/GridWorlds.jl#list-of-environments) | [GridWorlds.jl](https://github.com/JuliaReinforcementLearning/GridWorlds.jl) | Environments in this package use the interfaces defined in `RLBae` directly |

## Usage

```julia
julia> using ReinforcementLearningEnvironments

julia> using ReinforcementLearningBase

julia> env = CartPoleEnv()
# CartPoleEnv

## Traits

| Trait Type        |                  Value |
|:----------------- | ----------------------:|
| NumAgentStyle     |          SingleAgent() |
| DynamicStyle      |           Sequential() |
| InformationStyle  | ImperfectInformation() |
| ChanceStyle       |           Stochastic() |
| RewardStyle       |           StepReward() |
| UtilityStyle      |           GeneralSum() |
| ActionStyle       |     MinimalActionSet() |
| StateStyle        |     Observation{Any}() |
| DefaultStateStyle |     Observation{Any}() |

## Is Environment Terminated?

No

## State Space

`Space{Array{IntervalSets.Interval{:closed,:closed,Float64},1}}(IntervalSets.Interval{:closed,:closed,Float64}[-4.8..4.8, -1.0e38..1.0e38, -0.41887902047863906..0.41887902047863906, -1.0e38..1.0e38])`

## Action Space

`Base.OneTo(2)`

## Current State

[-0.032343893118127506, -0.04221525994544837, 0.024350079684957393, 0.04059943022508135]

julia> state(env)
4-element Array{Float64,1}:
  0.02688439956517477
 -0.0003235577964125977
  0.019563124862911535
 -0.01897808522860225

julia> action_space(env)
Base.OneTo(2)

julia> while !is_terminated(env)
           env(rand(legal_action_space(A)))
       end

julia> using ArcadeLearningEnvironment  # to use 3rd-party environments, you need to manually install and use the dependency first

julia> env = AtariEnv("pong");
```

### Environment Wrappers

Many handy environment [wrappers](https://github.com/JuliaReinforcementLearning/ReinforcementLearningEnvironments.jl/blob/master/src/environments/wrappers) are provided to mimic the OOP style
manipulation.

- `ActionTransformedEnv`
- `DefaultStateStyleEnv`
- `MaxTimeoutEnv`
- `RewardOverriddenEnv`
- `StateCachedEnv`
- `StateOverriddenEnv`

## Application

Checkout
[experiments](https://github.com/JuliaReinforcementLearning/ReinforcementLearningZoo.jl/tree/master/src/experiments)
in
[ReinforcementLearningZoo.jl](https://github.com/JuliaReinforcementLearning/ReinforcementLearningZoo.jl)
for how to apply modern reinforcement learning algorithms to these environments. You may also want to read this [pluto notebook](https://juliareinforcementlearning.org/blog/how_to_write_a_customized_environment/) for how to write a customized environment.
