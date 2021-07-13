# ReinforcementLearning.jl Release Notes

## ReinforcementLearning.jl@v0.9.0

### ReinforcementLearningBase.jl

#### v0.9.5

- Add default `Base.:(==)` and `Base.hash` method for `AbstractEnv`. [#348](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/348)

### ReinforcementLearningCore.jl

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

#### v0.6.1

- Drop GR in RLEnvs and lazily load ploting functions.[#309](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/309), [#310](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl/pull/310)

#### v0.6.0

- Set `AcrobotEnv` into lazy loading to reduce the dependency of `OrdinaryDiffEq`.

### ReinforcementLearningExperiments.jl

#### v0.1.0