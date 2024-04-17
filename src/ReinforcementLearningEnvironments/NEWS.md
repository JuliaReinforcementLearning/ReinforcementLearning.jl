### ReinforcementLearningEnvironments.jl Release Notes

#### v0.8

- Transition to `RLCore.forward`, `RLBase.act!`, `RLBase.plan!` and `Base.push!` syntax instead of functional objects for hooks, policies and environments

#### v0.7.2

- Reduce allocations, improve performance of `RandomWalk1D`
- Add tests to `RandomWalk1D`
- Chase down JET.jl errors, fix
- Update `TicTacToeEnv` and `RockPaperScissorsEnv` to support new `MultiAgentPolicy` setup

#### v0.6.12

- Bugfix bug with `is_discrete_space` [#566](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/issues/566)

#### v0.6.11

- Bugfix of CartPoleEnv with keyword arguments

#### v0.6.10

- Bugfix of CartPoleEnv with Float32

#### v0.6.9

- Added a continuous option for CartPoleEnv [#543](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/543).

#### v0.6.8

- Support `action_space(::TicTacToeEnv, player)`.

#### v0.6.7

- Fixed bugs in plotting `MountainCarEnv` (#537)
- Implemented plotting for `PendulumEnv` (#537)

#### v0.6.6

- Bugfix with `ZeroTo` [#534](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/534)

#### v0.6.4

- Add `GraphShortestPathEnv`. [#445](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/445)

#### v0.6.3

- Add `StockTradingEnv` from the paper [Deep Reinforcement Learning for
  Automated Stock Trading: An Ensemble
  Strategy](https://github.com/AI4Finance-Foundation/FinRL-Trading).
  This environment is a good testbed for multi-continuous action space
  algorithms. [#428](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/428)

#### v0.6.2

- Add `SequentialEnv` environment wrapper to turn a simultaneous environment
  into a sequential one.

#### v0.6.1

- Drop GR in RLEnvs and lazily load ploting functions.[#309](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/309), [#310](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/310)

#### v0.6.0

- Set `AcrobotEnv` into lazy loading to reduce the dependency of `OrdinaryDiffEq`.