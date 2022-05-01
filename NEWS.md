# ReinforcementLearning.jl Release Notes

## ReinforcementLearning.jl@v0.10.0

### ReinforcementLearningExperiments.jl

#### v0.1.4

- Support `device_rng` in SAC [#606](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/606)

#### v0.1.3

- Test experiments on GPU by default [#549](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/549)

#### v0.1.2

- Added an experiment for DQN training on discrete `PendulumEnv` (#537)

### ReinforcementLearningEnvironments.jl

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

### ReinforcementLearningCore.jl

#### v0.8.11

- When sending a `CircularArrayBuffer` to GPU devices, convert `CircularArrayBuffer` into `CuArray` instead of the adapted `CircularArrayBuffer` of `CuArray`. [#606](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/606)

#### v0.8.10

- Update dependency of `CircularArrayBuffers` to `v0.1.9`. [#602](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/602)
- Add `CovGaussianNetwork`. [#597](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/597)
#### v0.8.8

- Fix warning about `vararg.data` in Julia@v1.7 [#560](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/560)

#### v0.8.7

- Make `GaussianNetwork` differentiable. [#549](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/549)

#### v0.8.6

- Fixed a bug [1] with the `DoOnExit` hook (#537)
- Added some convenience hooks for rendering rollout episodes (#537)

#### v0.8.5

- Fixed the method overwritten warning of `device` from `CUDA.jl`.

### ReinforcementLearningZoo.jl

#### v0.5.11

- Fix multi-dimension action space in TD3. [#624](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/issues/624)

#### v0.5.10

- Support `device_rng` in SAC [#606](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/606)

#### v0.5.7

- Fix warning about `vararg.data` in Julia@v1.7 [#560](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/560)

#### v0.5.6

- Make BC GPU compatible [#553](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/553)

#### v0.5.5

- Make most algorithms GPU compatible [#549](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/549)

#### v0.5.4

- Support `length` method for `VectorWSARTTrajectory`.

#### v0.5.3

- Revert part of the unexpected change of PPO in the last PR.

#### v0.5.2

- Fixed the bug with MaskedPPOTrajectory reported [here](https://discourse.julialang.org/t/using-ppopolicy-with-custom-environment-with-action-masking-in-reinforcementlearning-jl/69625/6)

#### v0.5.0

- Update the complete SAC implementation and modify some details based on the
  original paper. [#365](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/365)
- Add some extra keyword parameters for `BehaviorCloningPolicy` to use it
  online. [#390](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/390)

### ReinforcementLearningDatasets.jl

#### v0.1.0

- Add functionality for fetching d4rl datasets as an iterable DataSet. Credits: https://arxiv.org/abs/2004.07219
- This supports d4rl and d4rl-pybullet and Google Research DQN atari datasets.
- Uses DataDeps for data dependency management.
- This package also supports RL Unplugged Datasets.
- Support for [google-research/deep_ope](https://github.com/google-research/deep_ope) added.

## ReinforcementLearning.jl@v0.9.0

### ReinforcementLearningBase.jl

#### v0.9.6

- Implement `Base.:(==)` for `Space`. [#428](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/428)

#### v0.9.5

- Add default `Base.:(==)` and `Base.hash` method for `AbstractEnv`. [#348](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/348)

### ReinforcementLearningCore.jl

#### v0.8.3

- Add extra two optional keyword arguments (`min_σ` and `max_σ`) in
  `GaussianNetwork` to clip the output of `logσ`. [#428](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/428)

#### v0.8.2

- Add GaussianNetwork and DuelingNetwork into ReinforcementLearningCore.jl as general components. [#370](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/370)
- Export `WeightedSoftmaxExplorer`.
  [#382](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/382)

#### v0.8.1

- Minor bug & typo fixes

#### v0.8.0

- Removed `ResizeImage` preprocessor to reduce the dependency of `ImageTransformations`. 
- Show unicode plot at the end of an experiment in the `TotalRewardPerEpisode` hook.

### ReinforcementLearningZoo.jl

#### v0.4.1

- Make keyword argument `n_actions` in `TabularPolicy` optional. [#300](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/300)

#### v0.4.0

- Moved all the experiments into a new package `ReinforcementLearningExperiments.jl`. The related dependencies are also removed (`BSON.jl`, `StableRNGs.jl`, `TensorBoardLogger.jl`).

### ReinforcementLearningEnvironments.jl

#### v0.6.4-dev

- Add `GraphShortestPathEnv`. [#445](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/445)

#### v0.6.3

- Add `StockTradingEnv` from the paper [Deep Reinforcement Learning for
  Automated Stock Trading: An Ensemble
  Strategy](https://github.com/AI4Finance-LLC/Deep-Reinforcement-Learning-for-Automated-Stock-Trading-Ensemble-Strategy-ICAIF-2020).
  This environment is a good testbed for multi-continuous action space
  algorithms. [#428](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/428)

#### v0.6.2

- Add `SequentialEnv` environment wrapper to turn a simultaneous environment
  into a sequential one.

#### v0.6.1

- Drop GR in RLEnvs and lazily load ploting functions.[#309](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/309), [#310](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/310)

#### v0.6.0

- Set `AcrobotEnv` into lazy loading to reduce the dependency of `OrdinaryDiffEq`.

### ReinforcementLearningExperiments.jl

#### v0.1.0
