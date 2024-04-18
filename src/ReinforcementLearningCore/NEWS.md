# ReinforcementLearningCore.jl Release Notes

#### v0.15.2

- Make QBasedPolicy general for AbstractLearner s (#1069)

#### v0.15.1

- Fix MultiPlayer hook bugs
- Clarify that the correct `push!` syntax is `push!(hook, stage, policy, env)` or `push!(hook, stage, policy, env, player)`; `push!(hook)` or other permutations now error as not implemented.

#### v0.15.0

- First version released with ReinforcementLearning v0.11

#### v0.10.1

- Fix hook issue with 'extra' call; always run `push!` at end of episode, regardless of whether stopped or terminated

#### v0.10.0

- Transition to `RLCore.forward`, `RLBase.act!`, `RLBase.plan!` and `Base.push!` syntax instead of functional objects for hooks, policies and environments

#### v0.9.3

- Add back multi-agent support with `MultiAgentPolicy` and `MultiAgentHook`

#### v0.9.2

- Use correct Flux.stack function signature
- Reduce allocations, improve performance of `RandomPolicy`
- Chase down JET.jl errors, fix
- Add tests for `StopAfterStep`, `StopAfterEpisode`
- Add tests, improve performance of `RewardsPerEpisode`
- Refactor `Agent` for speedup

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
