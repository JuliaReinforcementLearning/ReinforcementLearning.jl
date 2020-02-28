# ReinforcementLearningEnvironments.jl

[ReinforcementLearningEnvironments.jl](https://github.com/JuliaReinforcementLearning/ReinforcementLearningEnvironments.jl) (**RLEnvs**) provides some very common environments (for example: CartPoleEnv) together with some wrappers for 3-rd party environments.

To use those extra environments with `ReinforcementLearning.jl`, you have to manually install those dependent packages listed [here](https://github.com/JuliaReinforcementLearning/ReinforcementLearningEnvironments.jl#3-rd-party-environments). And use it like this:

```julia-repl
julia> using ArcadeLearningEnvironment
┌ Warning: Package ReinforcementLearningEnvironments does not have ArcadeLearningEnvironment in its dependencies:
│ - If you have ReinforcementLearningEnvironments checked out for development and have
│   added ArcadeLearningEnvironment as a dependency but haven't updated your primary
│   environment's manifest file, try `Pkg.resolve()`.
│ - Otherwise you may need to report an issue with ReinforcementLearningEnvironments
└ Loading ArcadeLearningEnvironment into ReinforcementLearningEnvironments from project dependency, future warnings for ReinforcementLearningEnvironments are suppressed.

julia> using ReinforcementLearning.ReinforcementLearningEnvironments

julia> env = AtariEnv()
```

The warning above can be safely ignored. Especially notice that you need to execute `using ReinforcementLearning.ReinforcementLearningEnvironments`.

```@docs
CartPoleEnv
MountainCarEnv
PendulumEnv
PendulumNonInteractiveEnv
```