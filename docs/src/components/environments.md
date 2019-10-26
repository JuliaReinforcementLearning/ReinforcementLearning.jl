# Environments

This package relies on some interfaces provided by the [ReinforcementLearningEnvironments.jl](https://github.com/JuliaReinforcementLearning/ReinforcementLearningEnvironments.jl) (**RLEnvs**). For completeness, we will also give a short introduction to it here.

RLEnvs provides many interfaces similar to [OpenAI Gym](https://gym.openai.com/docs/). But also extends it a little bit to make things easier to interact with sync/async, single/multi agent environments.

Basically, an environment is a functional object which takes in an action and changes its internal state correspondly. For sync environments, both `env(action)` and `reset!(env)` should return `nothing`, and `observe(env)` should return an [`Observation`](@ref). For async environments, they should all return a task.

A specially kind of environment is [`WrappedEnv`](@ref).

```@docs
Observation
WrappedEnv
```

## Preprocessors

Following are some built-in preprocessors. Notice that preprocessors can be chained (like `Chain(p1, p2, ps...)`) to get a composed preprocessor.

```@docs
AbstractPreprocessor
FourierPreprocessor
PolynomialPreprocessor
TilingPreprocessor
ImageResize
ImageCrop
ImageResizeNearestNeighbour
```